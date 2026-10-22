/*
 * In-memory .dat -> realfft -> rednoise pipeline.
 *
 * This is a faithful port of PRESTO's realfft.c/fastffts.c and rednoise.c for
 * the default rednoise parameters (startwidth=6, endwidth=100, endfreq=6 Hz).
 * Nothing is written to disk: the resulting complex spectrum is handed to the
 * search through an FFTFile whose amplitudes live in the pipeline's buffer.
 *
 * Numerical details carried over deliberately:
 *   - the list is treated as n/2 complex points (even samples real, odd imag),
 *     forward FFTW single-precision with FFTW_ESTIMATE (fftwcall chooses
 *     ESTIMATE for transforms larger than 16384 points);
 *   - the Numerical-Recipes real-FFT unpacking loop, with double trigonometry
 *     and float storage;
 *   - PRESTO's fast inverse square root (invsqrtf), including its finite
 *     approximation error -- it is not 1/sqrtf;
 *   - PRESTO's Numerical-Recipes quickselect median (lower median for even n);
 *   - red-noise block bookkeeping, including the partial final block.
 */
#include "coherent_search.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef TWOPI
#define TWOPI 6.2831853071795864769252867665590057683943387987502
#endif

typedef struct { float r, i; } dfcomplex;

struct DatPipe {
    long long N;
    long long ncomplex;
    fftwf_plan plan;
    float *samples;       /* N floats, realfft works here in place */
    float *red;           /* N floats = N/2 complex (the rednoise output) */
    float *powbuf;
    int powcap;
    dfcomplex *oldbuf, *newbuf, *outbuf;
    int bufcap;
    int startwidth, endwidth;
    double endfreq;
};

/* ------------------------------------------------------------------ */
/* PRESTO invsqrtf (misc_utils.c): fast inverse square root + 1 step. */
/* ------------------------------------------------------------------ */
static float presto_invsqrtf(float x)
{
    union { float f; int32_t i; } tmp;
    tmp.f = x;
    tmp.i = 0x5f3759df - (tmp.i >> 1);
    float y = tmp.f;
    return y * (1.5f - 0.5f * x * y * y);
}

/* ------------------------------------------------------------------ */
/* PRESTO median (median.c): Numerical Recipes quickselect, destroys   */
/* the array.  Returns element (n-1)/2, the lower median for even n.   */
/* ------------------------------------------------------------------ */
#define ELEM_SWAP(a, b) do { float t_ = (a); (a) = (b); (b) = t_; } while (0)

static float presto_median(float arr[], int n)
{
    int low, high, median, middle, ll, hh;
    low = 0;
    high = n - 1;
    median = (low + high) / 2;
    for (;;) {
        if (high <= low) return arr[median];
        if (high == low + 1) {
            if (arr[low] > arr[high]) ELEM_SWAP(arr[low], arr[high]);
            return arr[median];
        }
        middle = (low + high) / 2;
        if (arr[middle] > arr[high]) ELEM_SWAP(arr[middle], arr[high]);
        if (arr[low] > arr[high]) ELEM_SWAP(arr[low], arr[high]);
        if (arr[middle] > arr[low]) ELEM_SWAP(arr[middle], arr[low]);
        ELEM_SWAP(arr[middle], arr[low + 1]);
        ll = low + 1;
        hh = high;
        for (;;) {
            do ll++; while (arr[low] > arr[ll]);
            do hh--; while (arr[hh] > arr[low]);
            if (hh < ll) break;
            ELEM_SWAP(arr[ll], arr[hh]);
        }
        ELEM_SWAP(arr[low], arr[hh]);
        if (hh <= median) low = ll;
        if (hh >= median) high = hh - 1;
    }
}

/* ------------------------------------------------------------------ */
/* realfft() forward (fastffts.c)                                     */
/* ------------------------------------------------------------------ */
static void realfft_forward(float *idata, long long n, fftwf_plan plan)
{
    long long nby2 = n >> 1;
    dfcomplex *data = (dfcomplex *)idata;
    const double cc = -0.5;
    const double theta = -TWOPI / (double)n;

    fftwf_execute(plan);        /* COMPLEXFFT(data, nby2, -1) */

    double wr = cos(theta);
    double wi = sin(theta);
    double tmp1 = sin(0.5 * theta);
    const double wpr = -2.0 * tmp1 * tmp1;
    const double wpi = wi;
    long long il = 1;
    long long ih = nby2 - il;
    for (; il <= (n >> 2); il++, ih--) {
        double h1r = 0.5 * (data[il].r + data[ih].r);
        double h1i = 0.5 * (data[il].i - data[ih].i);
        double h2r = -cc * (data[il].i + data[ih].i);
        double h2i = cc * (data[il].r - data[ih].r);
        double h2rwr = h2r * wr;
        double h2rwi = h2r * wi;
        double h2iwr = h2i * wr;
        double h2iwi = h2i * wi;
        data[il].r = (float)(h1r + h2rwr - h2iwi);
        data[il].i = (float)(h1i + h2iwr + h2rwi);
        data[ih].r = (float)(h1r - h2rwr + h2iwi);
        data[ih].i = (float)(-h1i + h2iwr + h2rwi);
        tmp1 = wr;
        wr = tmp1 * wpr - wi * wpi + wr;
        wi = wi * wpr + tmp1 * wpi + wi;
    }
    /* Set data[0].r to Freq 0, data[0].i to Nyquist */
    tmp1 = data[0].r;
    data[0].r = (float)(tmp1 + data[0].i);
    data[0].i = (float)(tmp1 - data[0].i);
}

/* ------------------------------------------------------------------ */
/* rednoise removal (rednoise.c)                                      */
/* ------------------------------------------------------------------ */
static int rednoise_apply(const dfcomplex *in, dfcomplex *out, long long ncomplex,
                          long long N, double dt, int startwidth, int endwidth,
                          double endfreq, float *powbuf,
                          dfcomplex *oldbuf, dfcomplex *newbuf,
                          dfcomplex *outbuf)
{
    long long binnum = 1, numwrote = 0;
    int bufflen, nblk_old, nblk_new, mid_old, mid_new, ii, ind;
    float mean_old, mean_new, T, dslope = 1.0f, norm;
    const float Tf = (float)((double)N * dt);
    T = Tf;

    /* DC / Nyquist bin is replaced by (1,0) exactly as PRESTO does. */
    out[0].r = 1.0f;
    out[0].i = 0.0f;
    numwrote = 1;

    bufflen = startwidth;
    if (ncomplex - 1 < bufflen) {
        fprintf(stderr, "Error: rednoise: too few Fourier bins (%lld)\n", ncomplex);
        return -1;
    }
    long long pos = 1;
    nblk_old = bufflen;
    memcpy(oldbuf, in + pos, (size_t)nblk_old * sizeof(dfcomplex));
    pos += nblk_old;
    mid_old = nblk_old / 2;

    for (ii = 0; ii < nblk_old; ii++) {
        float r = oldbuf[ii].r, im = oldbuf[ii].i;
        powbuf[ii] = r * r + im * im;
    }
    mean_old = (float)(presto_median(powbuf, nblk_old) / log(2.0));

    norm = presto_invsqrtf(mean_old);
    for (ii = 0; ii < mid_old; ii++) {
        outbuf[ii].r = oldbuf[ii].r * norm;
        outbuf[ii].i = oldbuf[ii].i * norm;
    }
    memcpy(out + numwrote, outbuf, (size_t)mid_old * sizeof(dfcomplex));
    numwrote += mid_old;

    binnum += nblk_old;
    bufflen = (int)(startwidth * log((double)binnum));

    for (;;) {
        long long avail = ncomplex - pos;
        if (avail <= 0) break;
        nblk_new = (avail < bufflen) ? (int)avail : bufflen;
        memcpy(newbuf, in + pos, (size_t)nblk_new * sizeof(dfcomplex));
        pos += nblk_new;
        mid_new = nblk_new / 2;

        for (ii = 0; ii < nblk_new; ii++) {
            float r = newbuf[ii].r, im = newbuf[ii].i;
            powbuf[ii] = r * r + im * im;
        }
        mean_new = (float)(presto_median(powbuf, nblk_new) / log(2.0));

        dslope = (float)((mean_new - mean_old) / (0.5 * (nblk_old + nblk_new)));

        ii = 0;
        for (ind = mid_old; ind < nblk_old; ii++, ind++) {
            norm = presto_invsqrtf(mean_old + dslope * (float)ii);
            outbuf[ii].r = oldbuf[ind].r * norm;
            outbuf[ii].i = oldbuf[ind].i * norm;
        }
        for (ind = 0; ind < mid_new; ii++, ind++) {
            norm = presto_invsqrtf(mean_old + dslope * (float)ii);
            outbuf[ii].r = newbuf[ind].r * norm;
            outbuf[ii].i = newbuf[ind].i * norm;
        }
        memcpy(out + numwrote, outbuf, (size_t)ii * sizeof(dfcomplex));
        numwrote += ii;

        binnum += nblk_new;
        if ((float)binnum / T < endfreq)
            bufflen = (int)(startwidth * log((double)binnum));
        else
            bufflen = endwidth;
        if (bufflen > endwidth) {
            fprintf(stderr, "Error: rednoise block length %d exceeds buffer %d\n",
                    bufflen, endwidth);
            return -1;
        }

        dfcomplex *tmp = oldbuf;
        oldbuf = newbuf;
        newbuf = tmp;
        nblk_old = nblk_new;
        mean_old = mean_new;
        mid_old = mid_new;

    }

    /* Last half block, using the most recent slope. */
    ii = 0;
    for (ind = mid_old; ind < nblk_old; ii++, ind++) {
        norm = presto_invsqrtf(mean_old + dslope * (float)ii);
        outbuf[ii].r = oldbuf[ind].r * norm;
        outbuf[ii].i = oldbuf[ind].i * norm;
    }
    memcpy(out + numwrote, outbuf, (size_t)ii * sizeof(dfcomplex));
    numwrote += ii;

    if (numwrote != ncomplex) {
        fprintf(stderr, "Error: rednoise wrote %lld of %lld bins\n", numwrote, ncomplex);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Pipeline object                                                    */
/* ------------------------------------------------------------------ */
DatPipe *datpipe_create(int startwidth, int endwidth, double endfreq)
{
    DatPipe *p = (DatPipe *)calloc(1, sizeof(DatPipe));
    if (!p) return NULL;
    p->startwidth = startwidth > 0 ? startwidth : 6;
    p->endwidth = endwidth > 0 ? endwidth : 100;
    p->endfreq = endfreq > 0.0 ? endfreq : 6.0;
    return p;
}

void datpipe_free(DatPipe *p)
{
    if (!p) return;
    cs_fftw_plan_lock();
    if (p->plan) fftwf_destroy_plan(p->plan);
    cs_fftw_plan_unlock();
    if (p->samples) fftwf_free(p->samples);
    free(p->red);
    free(p->powbuf);
    free(p->oldbuf);
    free(p->newbuf);
    free(p->outbuf);
    free(p);
}

static int datpipe_prepare(DatPipe *p, long long N)
{
    if (p->N == N && p->samples && p->red) return 0;
    cs_fftw_plan_lock();
    if (p->plan) { fftwf_destroy_plan(p->plan); p->plan = NULL; }
    cs_fftw_plan_unlock();
    if (p->samples) { fftwf_free(p->samples); p->samples = NULL; }
    free(p->red); p->red = NULL;
    free(p->powbuf); p->powbuf = NULL;
    free(p->oldbuf); free(p->newbuf); free(p->outbuf);
    p->oldbuf = p->newbuf = p->outbuf = NULL;

    if (N <= 0 || (N & 1)) return -1;
    if (N / 2 > (long long)INT_MAX) {
        fprintf(stderr, "Error: time series too long for a single FFTW plan (%lld)\n", N);
        return -1;
    }
    p->N = N;
    p->ncomplex = N / 2;
    p->samples = (float *)fftwf_malloc((size_t)N * sizeof(float));
    p->red = (float *)malloc((size_t)N * sizeof(float));
    if (!p->samples || !p->red) return -1;

    int nfft = (int)p->ncomplex;
    /* PRESTO's fftwcall builds the forward and inverse plans back to back and
     * then executes the forward one; doing the same reproduces its plan choice
     * (the forward-only sequence differs in the last bits).  The inverse plan
     * is never executed, so release it immediately instead of holding one per
     * pipeline slot for the whole run. */
    cs_fftw_plan_lock();
    p->plan = fftwf_plan_dft_1d(nfft, (fftwf_complex *)p->samples,
                                (fftwf_complex *)p->samples, FFTW_FORWARD,
                                FFTW_ESTIMATE);
    fftwf_plan inv = fftwf_plan_dft_1d(nfft, (fftwf_complex *)p->samples,
                                      (fftwf_complex *)p->samples, FFTW_BACKWARD,
                                      FFTW_ESTIMATE);
    cs_fftw_plan_unlock();
    if (!p->plan || !inv) return -1;
    cs_fftw_plan_lock();
    fftwf_destroy_plan(inv);
    cs_fftw_plan_unlock();

    int cap = p->endwidth;
    p->powcap = cap;
    p->bufcap = cap;
    p->powbuf = (float *)malloc((size_t)cap * sizeof(float));
    p->oldbuf = (dfcomplex *)malloc((size_t)cap * sizeof(dfcomplex));
    p->newbuf = (dfcomplex *)malloc((size_t)cap * sizeof(dfcomplex));
    p->outbuf = (dfcomplex *)malloc((size_t)cap * sizeof(dfcomplex));
    if (!p->powbuf || !p->oldbuf || !p->newbuf || !p->outbuf) return -1;
    return 0;
}

int datpipe_redfft(DatPipe *p, const char *datpath, FFTFile *ft)
{
    memset(ft, 0, sizeof(*ft));
    ft->path = strdup(datpath);
    if (!ft->path) return -1;

    size_t plen = strlen(datpath);
    size_t keep = plen > 4 ? plen - 4 : 0;
    char *infpath = (char *)malloc(keep + 5);
    if (!infpath) return -1;
    memcpy(infpath, datpath, keep);
    memcpy(infpath + keep, ".inf", 5);
    if (inf_read(infpath, &ft->inf) != 0) {
        fprintf(stderr, "Error: The .inf file '%s' was not found.\n", infpath);
        free(infpath);
        return -1;
    }
    free(infpath);
    if (!ft->inf.has_N || !ft->inf.has_dt) {
        fprintf(stderr, "Error: missing N or dt in %s\n", ft->inf.path);
        return -1;
    }
    long long N = ft->inf.N;
    if (datpipe_prepare(p, N) != 0) {
        fprintf(stderr, "Error: cannot prepare realfft/rednoise for N=%lld\n", N);
        return -1;
    }

    FILE *f = fopen(datpath, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open '%s'\n", datpath);
        return -1;
    }
    size_t got = fread(p->samples, sizeof(float), (size_t)N, f);
    fclose(f);
    if (got != (size_t)N) {
        fprintf(stderr, "Error: '%s' holds %zu floats, expected %lld\n",
                datpath, got, N);
        return -1;
    }

    realfft_forward(p->samples, N, p->plan);
    if (rednoise_apply((const dfcomplex *)p->samples, (dfcomplex *)p->red,
                       p->ncomplex, N, ft->inf.dt, p->startwidth, p->endwidth,
                       p->endfreq, p->powbuf, p->oldbuf, p->newbuf,
                       p->outbuf) != 0)
        return -1;

    ft->N = N;
    ft->T = (double)N * ft->inf.dt;
    ft->df = 1.0 / ft->T;
    ft->dereddened = 1;
    ft->detrended = 1;
    ft->amps = p->red;
    ft->namps = p->ncomplex;
    ft->map = NULL;
    ft->mapsize = 0;
    ft->owns_amps = 0;      /* owned by the pipeline */
    ft->DC = ft->amps[0];
    ft->Nyquist = ft->amps[1];
    return 0;
}

int cs_open_input(const char *path, DatPipe *pipe, FFTFile *ft, int *is_dat)
{
    *is_dat = 0;
    size_t n = strlen(path);
    if (n >= 4 && strcmp(path + n - 4, ".dat") == 0) {
        *is_dat = 1;
        if (!pipe) {
            fprintf(stderr, "Error: no .dat pipeline available\n");
            return -1;
        }
        return datpipe_redfft(pipe, path, ft);
    }
    if (n >= 4 && strcmp(path + n - 4, ".fft") == 0)
        return fftfile_open(ft, path);
    fprintf(stderr, "Error: input '%s' must end in .dat or .fft\n", path);
    return -1;
}
