/* Direct O(m) Fourier interpolation: plans, chunk filling, chunk splitting. */
#include "coherent_search.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

/* Julia-compatible sinpi/cospi. */
double cs_sinpi(double x)
{
    if (!isfinite(x)) return x - x;
    double r = fmod(x, 2.0);
    if (r < 0) r += 2.0;
    if (r == 0.0 || r == 1.0) return 0.0 * r; /* preserve signed zero like sinpi */
    if (r == 0.5) return 1.0;
    if (r == 1.5) return -1.0;
    return sin(M_PI * r);
}
double cs_cospi(double x)
{
    if (!isfinite(x)) return x - x;
    double r = fmod(x, 2.0);
    if (r < 0) r += 2.0;
    if (r == 0.0) return 1.0;
    if (r == 0.5 || r == 1.5) return 0.0;
    if (r == 1.0) return -1.0;
    return cos(M_PI * r);
}
/* Julia eps(x): ulp at x (distance to the next larger float). */
double cs_eps(double x)
{
    x = fabs(x);
    if (x == 0.0) return 4.9406564584124654e-324;
    return nextafter(x, INFINITY) - x;
}

/* ------------------------------------------------------------------ */
/* Parameters                                                         */
/* ------------------------------------------------------------------ */
void params_defaults(SearchParams *p)
{
    memset(p, 0, sizeof(*p));
    p->nharms = 60;
    p->m = 16;
    p->numbetween = 16;
    p->hidr = 0.5;
    p->threshold = 8.0;
    p->boxcar_fsp = 1.5;
    p->boxcar_maxfrac = 0.3;
    p->boxcar_gatemargin = 0.01;
    p->precision = CS_PREC_F32;
    p->sigma = CS_SIGMA_ANALYTIC;
    p->ndecim = decimation_set(p->nharms, 6, p->decimations, 64);
}

int decimation_set(int nharms, int maxdecim, int *out, int outcap)
{
    if (maxdecim < 1) maxdecim = 1;
    int n = 0;
    { int k; for (k = 1; k <= maxdecim; k++) {
        if (nharms / k >= 2) {
            if (n < outcap) out[n] = k;
            n++;
        }
    } }
    return n < outcap ? n : outcap;
}

/* ------------------------------------------------------------------ */
/* Rational trial grid                                                */
/* ------------------------------------------------------------------ */
int trial_grid_rational(double lodr, int *pnum_out, int *q_out)
{
    if (!(lodr > 0.0)) return 0;
    /* Continued-fraction rationalisation, like Julia's rationalize(Int, x; tol=1e-13). */
    const double tol = 1e-13;
    long long p0 = 0, q0 = 1, p1 = 1, q1 = 0;
    double y = lodr;
    long long p = 0, q = 1;
    int found = 0;
    { int it; for (it = 0; it < 200; it++) {
        double fl = floor(y);
        if (fl > 2.0e9) break;
        long long a = (long long)fl;
        __int128 pn = (__int128)a * p1 + p0;
        __int128 qn = (__int128)a * q1 + q0;
        if (pn > (__int128)1e18 || qn > (__int128)1e18) break;
        p = (long long)pn;
        q = (long long)qn;
        p0 = p1; q0 = q1; p1 = p; q1 = q;
        if (q > CS_DIRECT_QMAX) break;
        double approx = (double)p / (double)q;
        if (fabs(approx - lodr) <= tol) { found = 1; break; }
        double frac = y - fl;
        if (frac <= 0.0) break;
        y = 1.0 / frac;
        if (!isfinite(y)) break;
    } }
    if (!found) {
        /* also accept the last convergent if it matches to rounding */
        if (q > 0 && q <= CS_DIRECT_QMAX && p >= 1 &&
            fabs((double)p / (double)q - lodr) <= tol)
            found = 1;
    }
    if (!found || q <= 0 || q > CS_DIRECT_QMAX || p < 1) return 0;
    if (fabs((double)p / (double)q - lodr) > 8.0 * cs_eps(lodr)) return 0;
    *pnum_out = (int)p;
    *q_out = (int)q;
    return 1;
}

/* ------------------------------------------------------------------ */
/* Plan construction                                                  */
/* ------------------------------------------------------------------ */
static int gcd_int(int a, int b)
{
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) { int t = a % b; a = b; b = t; }
    return a;
}

static void plan_free(DirectPlan *p)
{
    free(p->row);
    free(p->W);
    free(p->A_re);
    free(p->A_im);
    free(p->carry);
    free(p->grow);
    free(p->goff);
    free(p->gnj);
    free(p->gW);
    free(p->gA_re);
    free(p->gA_im);
    memset(p, 0, sizeof(*p));
}

static int build_group_table(DirectPlan *dp)
{
    const int V = CS_DIRECT_GROUP_V;
    const int m = dp->m, q = dp->q, s = dp->s, base_adv = dp->base_adv;
    int gstep = (int)(((long long)V * s) % q);
    int ngrp = (gstep == 0) ? 1 : q / gcd_int(gstep, q);
    dp->ngrp = ngrp;
    dp->grow = (int32_t *)calloc((size_t)q, sizeof(int32_t));
    dp->goff = (int32_t *)malloc((size_t)(ngrp + 1) * sizeof(int32_t));
    dp->gnj = (int32_t *)malloc((size_t)ngrp * sizeof(int32_t));
    dp->gA_re = (float *)malloc((size_t)V * ngrp * sizeof(float));
    dp->gA_im = (float *)malloc((size_t)V * ngrp * sizeof(float));
    if (!dp->grow || !dp->goff || !dp->gnj || !dp->gA_re || !dp->gA_im) return -1;

    int *ps = (int *)malloc((size_t)V * sizeof(int));
    int *ds = (int *)malloc((size_t)V * sizeof(int));
    if (!ps || !ds) { free(ps); free(ds); return -1; }

    /* First pass computes group sizes; second fills the flat weight table. */
    size_t total = 0;
    int res0 = 0;
    { int g; for (g = 0; g < ngrp; g++) {
        dp->grow[res0] = (int32_t)(g + 1);
        int res = res0, B = 0, B0 = 0;
        { int k; for (k = 0; k < V; k++) {
            int p = (int)dp->row[res] - 1;
            int Bk = B + (dp->carry[p] ? 1 : 0);
            if (k == 0) B0 = Bk;
            ps[k] = p;
            ds[k] = Bk - B0;
            dp->gA_re[(size_t)g * V + k] = dp->A_re[p];
            dp->gA_im[(size_t)g * V + k] = dp->A_im[p];
            res += s;
            B += base_adv;
            if (res >= q) { res -= q; B += 1; }
        } }
        int nj = m + ds[V - 1];
        dp->gnj[g] = (int32_t)nj;
        dp->goff[g] = (int32_t)total;
        total += (size_t)V * nj;
        res0 = (res0 + gstep) % q;
    } }
    dp->goff[ngrp] = (int32_t)total;
    dp->gW = (float *)calloc(total ? total : 1, sizeof(float));
    if (!dp->gW) { free(ps); free(ds); return -1; }

    res0 = 0;
    { int g; for (g = 0; g < ngrp; g++) {
        int res = res0, B = 0, B0 = 0;
        { int k; for (k = 0; k < V; k++) {
            int p = (int)dp->row[res] - 1;
            int Bk = B + (dp->carry[p] ? 1 : 0);
            if (k == 0) B0 = Bk;
            ps[k] = p;
            ds[k] = Bk - B0;
            res += s;
            B += base_adv;
            if (res >= q) { res -= q; B += 1; }
        } }
        size_t o = (size_t)dp->goff[g];
        { int k; for (k = 0; k < V; k++) {
            { int i; for (i = 0; i < m; i++) {
                dp->gW[o + (size_t)(ds[k] + i) * V + k] = dp->W[(size_t)ps[k] * m + i];
            } }
        } }
        res0 = (res0 + gstep) % q;
    } }
    free(ps);
    free(ds);
    return 0;
}

static int plan_build_one(DirectPlan *dp, int h, const SearchParams *p,
                          double r_lo, int has_rational, int pnum, int q)
{
    memset(dp, 0, sizeof(*dp));
    const int m = p->m;
    const int m2 = m / 2;
    dp->h = h;
    dp->m = m;
    dp->pnum = pnum;
    dp->q = q;
    double hr = (double)h * r_lo;
    dp->rfloor0 = (int)floor(hr);
    dp->dr0 = hr - (double)dp->rfloor0;

    if (!has_rational) {
        dp->P = 0;
        dp->s = 0;
        dp->base_adv = 0;
        return 0;
    }

    long long step = (long long)h * pnum;
    dp->s = (int)(step % q);
    dp->base_adv = (int)(step / q);
    /* Julia: P = q ÷ gcd(step, q). */
    {
        long long g = step, a = q;
        while (a) { long long t = g % a; g = a; a = t; }
        dp->P = (int)(q / g);
    }
    int P = dp->P;
    dp->row = (int32_t *)calloc((size_t)q, sizeof(int32_t));
    dp->W = (float *)malloc((size_t)m * P * sizeof(float));
    dp->A_re = (float *)malloc((size_t)P * sizeof(float));
    dp->A_im = (float *)malloc((size_t)P * sizeof(float));
    dp->carry = (uint8_t *)malloc((size_t)P);
    if (!dp->row || !dp->W || !dp->A_re || !dp->A_im || !dp->carry) return -1;

    int res = 0;
    { int pp; for (pp = 0; pp < P; pp++) {
        dp->row[res] = (int32_t)(pp + 1);
        double u = dp->dr0 + (double)res / (double)q;
        dp->carry[pp] = (u >= 1.0);
        double dr = dp->carry[pp] ? u - 1.0 : u;
        if (dr == 0.0) {
            dp->A_re[pp] = 1.0f;
            dp->A_im[pp] = 0.0f;
            { int i; for (i = 0; i < m; i++)
                dp->W[(size_t)pp * m + i] = (i == m2 - 1) ? 1.0f : 0.0f; }
        } else {
            double s_ = cs_sinpi(dr), c_ = cs_cospi(dr);
            dp->A_re[pp] = (float)(s_ * c_ / M_PI);
            dp->A_im[pp] = (float)(-s_ * s_ / M_PI);
            { int i; for (i = 0; i < m; i++) {
                /* i is 0-based; Julia's i=1..m has (i - m2) */
                dp->W[(size_t)pp * m + i] = (float)(1.0 / (dr - (double)(i + 1 - m2)));
            } }
        }
        res += dp->s;
        if (res >= q) res -= q;
    } }
    return build_group_table(dp);
}

int direct_plans_build(DirectPlanSet *set, const SearchParams *p, double r_lo)
{
    memset(set, 0, sizeof(*set));
    if (p->m <= 0 || (p->m & 1)) {
        fprintf(stderr, "Error: m must be a positive even integer\n");
        return -1;
    }
    double lodr = p->hidr / (double)p->nharms;
    int pnum = 0, q = 0;
    int has = trial_grid_rational(lodr, &pnum, &q);
    set->n = p->nharms;
    set->plans = (DirectPlan *)calloc((size_t)set->n, sizeof(DirectPlan));
    if (!set->plans) return -1;
    { int h; for (h = 1; h <= p->nharms; h++) {
        if (plan_build_one(&set->plans[h - 1], h, p, r_lo, has, pnum, q) != 0) {
            direct_plans_free(set);
            return -1;
        }
    } }
    return 0;
}

void direct_plans_free(DirectPlanSet *set)
{
    if (!set) return;
    { int i; for (i = 0; i < set->n; i++) plan_free(&set->plans[i]); }
    free(set->plans);
    set->plans = NULL;
    set->n = 0;
}

/* ------------------------------------------------------------------ */
/* Exact point interpolation (candidate profiles)                     */
/* ------------------------------------------------------------------ */
void fourier_interp_complex(double r, const float *amps, long long namps, int m,
                            double *out_re, double *out_im)
{
    const int m2 = m / 2;
    double dr = fmod(r, 1.0);
    if (dr < 0) dr += 1.0;
    long long rint0 = (long long)floor(r + 1e-15) + 1; /* 1-based */
    long long lo0 = rint0 - m2;                       /* 0-based first amp */
    double sre = 0.0, sim = 0.0;
    { int j; for (j = 0; j < m; j++) {
        double off = dr - (double)(-m2 + 1 + j);
        double sinc;
        if (off == 0.0) sinc = 1.0;
        else sinc = sin(M_PI * off) / (M_PI * off);
        double c = cos(M_PI * off), s = sin(M_PI * off);
        double cr = sinc * c, ci = -sinc * s; /* conj(coeff) */
        long long idx = lo0 + j;
        double bre = 0.0, bim = 0.0;
        if (idx >= 0 && idx < namps) {
            bre = (double)amps[2 * idx];
            bim = (double)amps[2 * idx + 1];
        }
        sre += cr * bre - ci * bim;
        sim += cr * bim + ci * bre;
    } }
    *out_re = sre;
    *out_im = sim;
}

/* ------------------------------------------------------------------ */
/* Chunk state / availability                                         */
/* ------------------------------------------------------------------ */
static inline void chunk_state(const DirectPlan *dp, long long t,
                               int *res, long long *qint)
{
    __int128 prod = (__int128)t * (long long)(dp->h * dp->pnum);
    long long q = dp->q;
    long long rem = (long long)(prod % q);
    if (rem < 0) rem += q;
    *res = (int)rem;
    *qint = (long long)(prod / q);
}

static inline int plan_base_at(const DirectPlan *dp, const SearchParams *p,
                               long long t)
{
    if (dp->P == 0) {
        double r = (double)(dp->rfloor0 + dp->dr0) +
                   (double)t * ((double)dp->h * (p->hidr / (double)p->nharms));
        return (int)floor(r + 1e-15);
    }
    int res; long long qint;
    chunk_state(dp, t, &res, &qint);
    int pidx = (int)dp->row[res] - 1;
    return (int)((long long)dp->rfloor0 + qint + (dp->carry[pidx] ? 1 : 0));
}

static inline int plan_nyq_at(const DirectPlan *dp, const SearchParams *p, long long t)
{
    if (dp->P == 0) {
        double r = (double)(dp->rfloor0 + dp->dr0) +
                   (double)t * ((double)dp->h * (p->hidr / (double)p->nharms));
        return (int)floor(r);
    }
    int res; long long qint;
    chunk_state(dp, t, &res, &qint);
    return (int)((long long)dp->rfloor0 + qint);
}

static inline int harmonic_fits(const DirectPlan *dp, const FFTFile *ft,
                                const SearchParams *p, long long t)
{
    const int m2 = dp->m / 2;
    const long long Nhalf = ft->N / 2;
    const long long namps = ft->namps;
    if (dp->P == 0) {
        double r = (double)(dp->rfloor0 + dp->dr0) +
                   (double)t * ((double)dp->h * (p->hidr / (double)p->nharms));
        long long rf = (long long)floor(r + 1e-15);
        return (rf + 2 - m2) >= 1 && (rf + 1 + m2) <= namps &&
               (long long)floor(r) < Nhalf;
    }
    int res; long long qint;
    chunk_state(dp, t, &res, &qint);
    int pidx = (int)dp->row[res] - 1;
    long long base = (long long)dp->rfloor0 + qint + (dp->carry[pidx] ? 1 : 0);
    return (base + 2 - m2) >= 1 && (base + 1 + m2) <= namps &&
           ((long long)dp->rfloor0 + qint) < Nhalf;
}

void cs_chunk_state(const DirectPlan *dp, long long t, int *res, long long *qint)
{
    chunk_state(dp, t, res, qint);
}

int cs_harmonic_fits(const DirectPlan *dp, const FFTFile *ft,
                     const SearchParams *p, long long t)
{
    return harmonic_fits(dp, ft, p, t);
}

int direct_window_size(const SearchParams *p, int Nprof)
{
    return (int)ceil((double)(Nprof - 1 + 2 * CS_DIRECT_GROUP_V) * p->hidr) +
           p->m + 4;
}

/* ------------------------------------------------------------------ */
/* Row filling                                                        */
/* ------------------------------------------------------------------ */
static int fill_row_slow(Workspace *ws, const DirectPlan *dp, const FFTFile *ft,
                         const SearchParams *p, long long t0, int n)
{
    const int h = dp->h, m = dp->m, m2 = m / 2;
    double lodr = p->hidr / (double)p->nharms;
    double hr_lo = (double)(dp->rfloor0 + dp->dr0);
    double dh = (double)h * lodr;
    const long long namps = ft->namps;
    if (!(harmonic_fits(dp, ft, p, t0) && harmonic_fits(dp, ft, p, t0 + n - 1)))
        return 0;
    { int k; for (k = 0; k < n; k++) {
        double r = hr_lo + (double)(t0 + k) * dh;
        long long rint0 = (long long)floor(r + 1e-15) + 1;
        double dr = fmod(r, 1.0);
        if (dr < 0) dr += 1.0;
        long long base = rint0 - m2; /* 1-based */
        double xr, xi;
        if (dr == 0.0) {
            long long idx = base + m2 - 1;
            xr = (idx >= 0 && idx < namps) ? (double)ft->amps[2 * idx] : 0.0;
            xi = (idx >= 0 && idx < namps) ? (double)ft->amps[2 * idx + 1] : 0.0;
        } else {
            double sre = 0.0, sim = 0.0;
            { int i; for (i = 1; i <= m; i++) {
                double w = 1.0 / (dr - (double)(i - m2));
                long long idx = base + i - 1;
                double ar = (idx >= 0 && idx < namps) ? (double)ft->amps[2 * idx] : 0.0;
                double ai = (idx >= 0 && idx < namps) ? (double)ft->amps[2 * idx + 1] : 0.0;
                sre += w * ar;
                sim += w * ai;
            } }
            double s_ = cs_sinpi(dr), c_ = cs_cospi(dr);
            double pr = s_ * c_ / M_PI;
            double pi_ = -s_ * s_ / M_PI;
            xr = pr * sre - pi_ * sim;
            xi = pr * sim + pi_ * sre;
        }
        if (ws->is_f64) {
            double *dst = (double *)ws->ftprofs;
            dst[2 * ((size_t)h * ws->Nprof + k)] = xr;
            dst[2 * ((size_t)h * ws->Nprof + k) + 1] = xi;
        } else {
            float *dst = (float *)ws->ftprofs;
            dst[2 * ((size_t)h * ws->Nprof + k)] = (float)xr;
            dst[2 * ((size_t)h * ws->Nprof + k) + 1] = (float)xi;
        }
    } }
    return 1;
}

/* Copy one harmonic's amplitude window into the workspace's planar re/im
 * buffers, zero-padding outside the stored spectrum.  Pure data movement. */
static int fill_planes_f32(Workspace *ws, const FFTFile *ft,
                           long long lo1, long long hi1, long long nw)
{
    const long long namps = ft->namps;
    float *re = ws->re, *im = ws->im;
    long long ihead = 1 - lo1;
    if (ihead < 0) ihead = 0;
    if (ihead > nw) ihead = nw;
    long long itail = hi1 - namps;
    if (itail < 0) itail = 0;
    if (itail > nw - ihead) itail = nw - ihead;

    { long long i; for (i = 0; i < ihead; i++) { re[i] = 0.0f; im[i] = 0.0f; } }
    long long mid_end = nw - itail;
    { long long i = ihead;
#if defined(__AVX512F__)
      /* De-interleave 16 complex at a time.  Pure data movement, so the
       * planes hold exactly what the scalar copy produced. */
      const __m512i ev = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14,
                                           16, 18, 20, 22, 24, 26, 28, 30);
      const __m512i od = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15,
                                           17, 19, 21, 23, 25, 27, 29, 31);
      for (; i + 16 <= mid_end; i += 16) {
          const float *ap = ft->amps + 2 * (lo1 + i - 1);
          __m512 a0 = _mm512_loadu_ps(ap);
          __m512 a1 = _mm512_loadu_ps(ap + 16);
          _mm512_storeu_ps(re + i, _mm512_permutex2var_ps(a0, ev, a1));
          _mm512_storeu_ps(im + i, _mm512_permutex2var_ps(a0, od, a1));
      }
#endif
      for (; i < mid_end; i++) {
          long long idx = lo1 + i - 1; /* 0-based amplitude */
          re[i] = ft->amps[2 * idx];
          im[i] = ft->amps[2 * idx + 1];
      } }
    { long long i; for (i = mid_end; i < nw; i++) { re[i] = 0.0f; im[i] = 0.0f; } }
    return 0;
}

/* Rotate the V accumulated complex values and store them into harmonic row h
 * of the h-major spectrum buffer (element (h, j) at h*Nprof + j), so each call
 * writes one contiguous run of V complex values.  Shared by the single-file
 * and batched fillers so both use exactly the same arithmetic and stores. */
static inline void row_store(Workspace *ws, int h, int nh1, long long j0, int V,
                             const float *ar, const float *ai, int g,
                             const float *gAr, const float *gAi, int n)
{
    float *ftf = (float *)ws->ftprofs;
    double *ftd = (double *)ws->ftprofs;
    const int Np = ws->Nprof;
    int full = (j0 >= 0 && j0 + V <= n);
    if (full) {
        int k = 0;
#if defined(__AVX512F__) && CS_DIRECT_GROUP_V == 32
        if (!ws->is_f64) {
            /* f32: rotate, interleave re/im across the whole register (a
             * plain unpack works per 128-bit lane and would permute trials
             * 4-7 with 8-11) and store 32 contiguous floats per pair. */
            const __m512i ev = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19,
                                                 4, 20, 5, 21, 6, 22, 7, 23);
            const __m512i od = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27,
                                                 12, 28, 13, 29, 14, 30, 15, 31);
            for (; k + 16 <= V; k += 16) {
                __m512 grv = _mm512_loadu_ps(gAr + (size_t)g * V + k);
                __m512 giv = _mm512_loadu_ps(gAi + (size_t)g * V + k);
                __m512 arv = _mm512_loadu_ps(ar + k);
                __m512 aiv = _mm512_loadu_ps(ai + k);
                __m512 xr = _mm512_sub_ps(_mm512_mul_ps(grv, arv),
                                          _mm512_mul_ps(giv, aiv));
                __m512 xi = _mm512_add_ps(_mm512_mul_ps(grv, aiv),
                                          _mm512_mul_ps(giv, arv));
                float *dst = ftf + 2 * ((size_t)h * Np + j0 + k);
                _mm512_storeu_ps(dst, _mm512_permutex2var_ps(xr, ev, xi));
                _mm512_storeu_ps(dst + 16, _mm512_permutex2var_ps(xr, od, xi));
            }
        } else {
            /* f64: the products are formed in f32 and widened per element,
             * exactly like the scalar tail (ftd[..] = (double)xr). */
            const __m512i ev = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
            const __m512i od = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
            for (; k + 8 <= V; k += 8) {
                __m256 grv = _mm256_loadu_ps(gAr + (size_t)g * V + k);
                __m256 giv = _mm256_loadu_ps(gAi + (size_t)g * V + k);
                __m256 arv = _mm256_loadu_ps(ar + k);
                __m256 aiv = _mm256_loadu_ps(ai + k);
                __m256 xr = _mm256_sub_ps(_mm256_mul_ps(grv, arv),
                                          _mm256_mul_ps(giv, aiv));
                __m256 xi = _mm256_add_ps(_mm256_mul_ps(grv, aiv),
                                          _mm256_mul_ps(giv, arv));
                __m512d xrd = _mm512_cvtps_pd(xr);
                __m512d xid = _mm512_cvtps_pd(xi);
                double *dst = ftd + 2 * ((size_t)h * Np + j0 + k);
                _mm512_storeu_pd(dst, _mm512_permutex2var_pd(xrd, ev, xid));
                _mm512_storeu_pd(dst + 8, _mm512_permutex2var_pd(xrd, od, xid));
            }
        }
#endif
        for (; k < V; k++) {
            float gr = gAr[(size_t)g * V + k];
            float gi = gAi[(size_t)g * V + k];
            float xr = gr * ar[k] - gi * ai[k];
            float xi = gr * ai[k] + gi * ar[k];
            if (ws->is_f64) {
                ftd[2 * ((size_t)h * Np + j0 + k)] = (double)xr;
                ftd[2 * ((size_t)h * Np + j0 + k) + 1] = (double)xi;
            } else {
                ftf[2 * ((size_t)h * Np + j0 + k)] = xr;
                ftf[2 * ((size_t)h * Np + j0 + k) + 1] = xi;
            }
        }
    } else {
        { int k; for (k = 0; k < V; k++) {
            long long j = j0 + k;
            if (j < 0 || j >= n) continue;
            float gr = gAr[(size_t)g * V + k];
            float gi = gAi[(size_t)g * V + k];
            float xr = gr * ar[k] - gi * ai[k];
            float xi = gr * ai[k] + gi * ar[k];
            if (ws->is_f64) {
                ftd[2 * ((size_t)h * Np + j)] = (double)xr;
                ftd[2 * ((size_t)h * Np + j) + 1] = (double)xi;
            } else {
                ftf[2 * ((size_t)h * Np + j)] = xr;
                ftf[2 * ((size_t)h * Np + j) + 1] = xi;
            }
        } }
    }
}

static int fill_row_group(Workspace *ws, const DirectPlan *dp, const FFTFile *ft,
                          const SearchParams *p, long long t0, int n)
{
    const int h = dp->h, m = dp->m, m2 = m / 2;
    const int V = CS_DIRECT_GROUP_V;
    const int nh1 = p->nharms + 1;

    if (!(harmonic_fits(dp, ft, p, t0) && harmonic_fits(dp, ft, p, t0 + n - 1)))
        return 0;

    long long T_first = (t0 / V) * V;
    long long T_last = ((t0 + n - 1) / V) * V + (V - 1);

    int res_e; long long qint_e;
    chunk_state(dp, T_first, &res_e, &qint_e);
    long long lo1 = (long long)dp->rfloor0 + qint_e +
                    (dp->carry[(int)dp->row[res_e] - 1] ? 1 : 0) + 2 - m2;
    int res_x; long long qint_x;
    chunk_state(dp, T_last, &res_x, &qint_x);
    long long hi1 = (long long)dp->rfloor0 + qint_x +
                    (dp->carry[(int)dp->row[res_x] - 1] ? 1 : 0) + 1 + m2;

    long long nw = hi1 - lo1 + 1;
    if (nw < 1) return 0;
    if ((size_t)nw > ws->planesize) {
        fprintf(stderr,
                "Error: workspace plane buffers hold %zu bins but harmonic %d spans "
                "%lld for a chunk of %d\n", ws->planesize, h, nw, n);
        return 0;
    }
    if (fill_planes_f32(ws, ft, lo1, hi1, nw) != 0) return 0;
    float *re = ws->re, *im = ws->im;

    const float *gW = dp->gW;
    const float *gAr = dp->gA_re, *gAi = dp->gA_im;
    const int32_t *grow = dp->grow, *goff = dp->goff, *gnj = dp->gnj;
    int res = res_e;
    long long qint = qint_e;
    long long T = T_first;

    while (T <= T_last) {
        /* Two groups per pass: the tap loop runs eight independent FMA
         * chains instead of four, covering the 4-cycle FMA latency without
         * extra passes over the weight rows.  Each group's FMA order is
         * unchanged, so the results are bit-identical. */
        long long T2 = T + V;
        int have2 = (T2 <= T_last);

        int gA = (int)grow[res] - 1;
        int pidxA = (int)dp->row[res] - 1;
        long long b0A = (long long)dp->rfloor0 + qint +
                        (dp->carry[pidxA] ? 1 : 0) + 2 - m2 - lo1;
        size_t oA = (size_t)goff[gA];
        int njA = (int)gnj[gA];
        long long j0A = T - t0;

        int resB = res;
        long long qintB = qint;
        int gB = 0, njB = 0;
        long long b0B = 0, j0B = 0;
        size_t oB = 0;
        if (have2) {
            long long totB = (long long)res + (long long)V * dp->s;
            qintB = qint + (long long)V * dp->base_adv + totB / dp->q;
            resB = (int)(totB % dp->q);
            gB = (int)grow[resB] - 1;
            int pidxB = (int)dp->row[resB] - 1;
            b0B = (long long)dp->rfloor0 + qintB +
                  (dp->carry[pidxB] ? 1 : 0) + 2 - m2 - lo1;
            oB = (size_t)goff[gB];
            njB = (int)gnj[gB];
            j0B = T2 - t0;
        }

        float ar[CS_DIRECT_GROUP_V];
        float ai[CS_DIRECT_GROUP_V];
        float ar2[CS_DIRECT_GROUP_V];
        float ai2[CS_DIRECT_GROUP_V];
#if defined(__AVX512F__) && CS_DIRECT_GROUP_V == 32
        __m512 arA0 = _mm512_setzero_ps(), arA1 = _mm512_setzero_ps();
        __m512 aiA0 = _mm512_setzero_ps(), aiA1 = _mm512_setzero_ps();
        __m512 arB0 = _mm512_setzero_ps(), arB1 = _mm512_setzero_ps();
        __m512 aiB0 = _mm512_setzero_ps(), aiB1 = _mm512_setzero_ps();
        int njm = have2 ? (njA < njB ? njA : njB) : njA;
        { int j; for (j = 0; j < njm; j++) {
            __m512 rv = _mm512_set1_ps(re[b0A + j]);
            __m512 iv = _mm512_set1_ps(im[b0A + j]);
            const float *w = gW + oA + (size_t)j * V;
            __m512 w0 = _mm512_loadu_ps(w + 0);
            __m512 w1 = _mm512_loadu_ps(w + 16);
            arA0 = _mm512_fmadd_ps(w0, rv, arA0);
            arA1 = _mm512_fmadd_ps(w1, rv, arA1);
            aiA0 = _mm512_fmadd_ps(w0, iv, aiA0);
            aiA1 = _mm512_fmadd_ps(w1, iv, aiA1);
            __m512 rv2 = _mm512_set1_ps(re[b0B + j]);
            __m512 iv2 = _mm512_set1_ps(im[b0B + j]);
            const float *w2 = gW + oB + (size_t)j * V;
            __m512 u0 = _mm512_loadu_ps(w2 + 0);
            __m512 u1 = _mm512_loadu_ps(w2 + 16);
            arB0 = _mm512_fmadd_ps(u0, rv2, arB0);
            arB1 = _mm512_fmadd_ps(u1, rv2, arB1);
            aiB0 = _mm512_fmadd_ps(u0, iv2, aiB0);
            aiB1 = _mm512_fmadd_ps(u1, iv2, aiB1);
        } }
        /* tail of the longer group (njA == njB for all but boundary groups) */
        { int j; for (j = njm; j < njA; j++) {
            __m512 rv = _mm512_set1_ps(re[b0A + j]);
            __m512 iv = _mm512_set1_ps(im[b0A + j]);
            const float *w = gW + oA + (size_t)j * V;
            __m512 w0 = _mm512_loadu_ps(w + 0);
            __m512 w1 = _mm512_loadu_ps(w + 16);
            arA0 = _mm512_fmadd_ps(w0, rv, arA0);
            arA1 = _mm512_fmadd_ps(w1, rv, arA1);
            aiA0 = _mm512_fmadd_ps(w0, iv, aiA0);
            aiA1 = _mm512_fmadd_ps(w1, iv, aiA1);
        } }
        if (have2) { int j; for (j = njm; j < njB; j++) {
            __m512 rv = _mm512_set1_ps(re[b0B + j]);
            __m512 iv = _mm512_set1_ps(im[b0B + j]);
            const float *w = gW + oB + (size_t)j * V;
            __m512 w0 = _mm512_loadu_ps(w + 0);
            __m512 w1 = _mm512_loadu_ps(w + 16);
            arB0 = _mm512_fmadd_ps(w0, rv, arB0);
            arB1 = _mm512_fmadd_ps(w1, rv, arB1);
            aiB0 = _mm512_fmadd_ps(w0, iv, aiB0);
            aiB1 = _mm512_fmadd_ps(w1, iv, aiB1);
        } }
        _mm512_storeu_ps(ar + 0, arA0); _mm512_storeu_ps(ar + 16, arA1);
        _mm512_storeu_ps(ai + 0, aiA0); _mm512_storeu_ps(ai + 16, aiA1);
        if (have2) {
            _mm512_storeu_ps(ar2 + 0, arB0); _mm512_storeu_ps(ar2 + 16, arB1);
            _mm512_storeu_ps(ai2 + 0, aiB0); _mm512_storeu_ps(ai2 + 16, aiB1);
        }
#elif defined(__AVX2__) && CS_DIRECT_GROUP_V == 32
        __m256 ar0 = _mm256_setzero_ps(), ar1 = _mm256_setzero_ps();
        __m256 ar2 = _mm256_setzero_ps(), ar3 = _mm256_setzero_ps();
        __m256 ai0 = _mm256_setzero_ps(), ai1 = _mm256_setzero_ps();
        __m256 ai2 = _mm256_setzero_ps(), ai3 = _mm256_setzero_ps();
        { int j; for (j = 0; j < njA; j++) {
            __m256 rv = _mm256_set1_ps(re[b0A + j]);
            __m256 iv = _mm256_set1_ps(im[b0A + j]);
            const float *w = gW + oA + (size_t)j * V;
            ar0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 0), rv, ar0);
            ar1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 8), rv, ar1);
            ar2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 16), rv, ar2);
            ar3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 24), rv, ar3);
            ai0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 0), iv, ai0);
            ai1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 8), iv, ai1);
            ai2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 16), iv, ai2);
            ai3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 24), iv, ai3);
        } }
        _mm256_storeu_ps(ar + 0, ar0); _mm256_storeu_ps(ar + 8, ar1);
        _mm256_storeu_ps(ar + 16, ar2); _mm256_storeu_ps(ar + 24, ar3);
        _mm256_storeu_ps(ai + 0, ai0); _mm256_storeu_ps(ai + 8, ai1);
        _mm256_storeu_ps(ai + 16, ai2); _mm256_storeu_ps(ai + 24, ai3);
        if (have2) {
            __m256 br0 = _mm256_setzero_ps(), br1 = _mm256_setzero_ps();
            __m256 br2 = _mm256_setzero_ps(), br3 = _mm256_setzero_ps();
            __m256 bi0 = _mm256_setzero_ps(), bi1 = _mm256_setzero_ps();
            __m256 bi2 = _mm256_setzero_ps(), bi3 = _mm256_setzero_ps();
            { int j; for (j = 0; j < njB; j++) {
                __m256 rv = _mm256_set1_ps(re[b0B + j]);
                __m256 iv = _mm256_set1_ps(im[b0B + j]);
                const float *w = gW + oB + (size_t)j * V;
                br0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 0), rv, br0);
                br1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 8), rv, br1);
                br2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 16), rv, br2);
                br3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 24), rv, br3);
                bi0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 0), iv, bi0);
                bi1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 8), iv, bi1);
                bi2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 16), iv, bi2);
                bi3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 24), iv, bi3);
            } }
            _mm256_storeu_ps(ar2 + 0, br0); _mm256_storeu_ps(ar2 + 8, br1);
            _mm256_storeu_ps(ar2 + 16, br2); _mm256_storeu_ps(ar2 + 24, br3);
            _mm256_storeu_ps(ai2 + 0, bi0); _mm256_storeu_ps(ai2 + 8, bi1);
            _mm256_storeu_ps(ai2 + 16, bi2); _mm256_storeu_ps(ai2 + 24, bi3);
        }
#else
        memset(ar, 0, sizeof(ar));
        memset(ai, 0, sizeof(ai));
        { int j; for (j = 0; j < njA; j++) {
            float rv = re[b0A + j];
            float iv = im[b0A + j];
            const float *w = gW + oA + (size_t)j * V;
            { int k; for (k = 0; k < V; k++) {
                ar[k] = fmaf(w[k], rv, ar[k]);
                ai[k] = fmaf(w[k], iv, ai[k]);
            } }
        } }
        if (have2) {
            memset(ar2, 0, sizeof(ar2));
            memset(ai2, 0, sizeof(ai2));
            { int j; for (j = 0; j < njB; j++) {
                float rv = re[b0B + j];
                float iv = im[b0B + j];
                const float *w = gW + oB + (size_t)j * V;
                { int k; for (k = 0; k < V; k++) {
                    ar2[k] = fmaf(w[k], rv, ar2[k]);
                    ai2[k] = fmaf(w[k], iv, ai2[k]);
                } }
            } }
        }
#endif

        row_store(ws, h, nh1, j0A, V, ar, ai, gA, gAr, gAi, n);
        if (have2)
            row_store(ws, h, nh1, j0B, V, ar2, ai2, gB, gAr, gAi, n);

        long long step = have2 ? 2 : 1;
        long long tot = (long long)res + (long long)step * V * dp->s;
        qint += (long long)step * V * dp->base_adv + tot / dp->q;
        res = (int)(tot % dp->q);
        T += (long long)step * V;
    }
    return 1;
}

int fill_harmonic_row(Workspace *ws, const DirectPlan *dp, const FFTFile *ft,
                      const SearchParams *p, long long t0, int n)
{
    if (n < 1) return 0;
    if (dp->P == 0) return fill_row_slow(ws, dp, ft, p, t0, n);
    return fill_row_group(ws, dp, ft, p, t0, n);
}

/* Zero one harmonic row (or the DC row, h = 0) of the h-major spectrum for
 * trials [0, n): a single contiguous run per row. */
static inline void zero_spec_row(Workspace *ws, int h, int n)
{
    char *base = (char *)ws->ftprofs;
    size_t csz = ws->is_f64 ? 2 * sizeof(double) : 2 * sizeof(float);
    memset(base + (size_t)h * ws->Nprof * csz, 0, (size_t)n * csz);
}

void fill_chunk_profiles(Workspace *ws, const DirectPlanSet *dplans,
                         const FFTFile *ft, const SearchParams *p,
                         double rstart, double lodr, int n, long long t0)
{
    (void)rstart; (void)lodr;
    /* Only the DC row and the harmonics that will NOT be filled need zeroing:
     * every filled harmonic writes all n columns, and columns >= n are never
     * read (the transforms and metrics only consume 0..n-1).  This replaces a
     * full (nharms+1)*Nprof memset per chunk. */
    zero_spec_row(ws, 0, n);
    { int i; for (i = 0; i < dplans->n; i++) {
        const DirectPlan *dp = &dplans->plans[i];
        int fits = harmonic_fits(dp, ft, p, t0) &&
                   harmonic_fits(dp, ft, p, t0 + n - 1);
        if (!fits) zero_spec_row(ws, dp->h, n);
        ws->filled[dp->h - 1] =
            (uint8_t)fill_harmonic_row(ws, dp, ft, p, t0, n);
    } }
}

/* ------------------------------------------------------------------ */
/* Harmonic availability change points                                */
/* ------------------------------------------------------------------ */
static int first_true(int (*pred)(const DirectPlan *, const SearchParams *, long long, void *),
                      const DirectPlan *dp, const SearchParams *p,
                      long long total, void *ctx)
{
    if (total < 1) return (int)total;
    if (pred(dp, p, 0, ctx)) return 0;
    if (!pred(dp, p, total - 1, ctx)) return (int)total;
    long long lo = 0, hi = total - 1;
    while (hi - lo > 1) {
        long long mid = (lo + hi) >> 1;
        if (pred(dp, p, mid, ctx)) hi = mid; else lo = mid;
    }
    return (int)hi;
}
static int first_false(int (*pred)(const DirectPlan *, const SearchParams *, long long, void *),
                       const DirectPlan *dp, const SearchParams *p,
                       long long total, void *ctx)
{
    if (total < 1) return (int)total;
    if (!pred(dp, p, 0, ctx)) return 0;
    if (pred(dp, p, total - 1, ctx)) return (int)total;
    long long lo = 0, hi = total - 1;
    while (hi - lo > 1) {
        long long mid = (lo + hi) >> 1;
        if (pred(dp, p, mid, ctx)) lo = mid; else hi = mid;
    }
    return (int)hi;
}

typedef struct { int m2; long long namps, Nhalf; } FitCtx;

static int pred_lo(const DirectPlan *dp, const SearchParams *p, long long t, void *v)
{
    FitCtx *c = (FitCtx *)v;
    return plan_base_at(dp, p, t) + 2 - c->m2 >= 1;
}
static int pred_hi(const DirectPlan *dp, const SearchParams *p, long long t, void *v)
{
    FitCtx *c = (FitCtx *)v;
    return plan_base_at(dp, p, t) + 1 + c->m2 <= c->namps &&
           plan_nyq_at(dp, p, t) < c->Nhalf;
}

static int ll_cmp(const void *a, const void *b)
{
    long long x = *(const long long *)a, y = *(const long long *)b;
    return (x > y) - (x < y);
}

int harmonic_change_trials(const DirectPlanSet *set, const FFTFile *ft,
                           const SearchParams *p, long long total,
                           long long **out_bnds)
{
    *out_bnds = NULL;
    if (total <= 1) return 0;
    int cap = 2 * set->n, n = 0;
    long long *b = (long long *)malloc((size_t)cap * sizeof(long long));
    if (!b) return 0;
    FitCtx ctx;
    ctx.namps = ft->namps;
    ctx.Nhalf = ft->N / 2;
    { int i; for (i = 0; i < set->n; i++) {
        const DirectPlan *dp = &set->plans[i];
        ctx.m2 = dp->m / 2;
        int a = first_true(pred_lo, dp, p, total, &ctx);
        int c = first_false(pred_hi, dp, p, total, &ctx);
        if (a > 0 && a < total) b[n++] = a;
        if (c > 0 && c < total) b[n++] = c;
    } }
    qsort(b, (size_t)n, sizeof(long long), ll_cmp);
    int m = 0;
    { int i; for (i = 0; i < n; i++) {
        if (m == 0 || b[i] != b[m - 1]) b[m++] = b[i];
    } }
    *out_bnds = b;
    return m;
}

int chunk_starts(const DirectPlanSet *set, const FFTFile *ft, const SearchParams *p,
                 long long total, int Nprof, long long **out, long long *nout)
{
    *out = NULL; *nout = 0;
    if (total < 1) return 0;
    long long nreg = (total + Nprof - 1) / Nprof;
    long long *bnds = NULL;
    int nb = harmonic_change_trials(set, ft, p, total, &bnds);
    long long cap = nreg + nb + 1;
    long long *s = (long long *)malloc((size_t)cap * sizeof(long long));
    if (!s) { free(bnds); return -1; }
    long long n = 0;
    { long long i; for (i = 0; i < total; i += Nprof) s[n++] = i; }
    { int i; for (i = 0; i < nb; i++) s[n++] = bnds[i]; }
    free(bnds);
    qsort(s, (size_t)n, sizeof(long long), ll_cmp);
    long long m = 0;
    { long long i; for (i = 0; i < n; i++)
        if (m == 0 || s[i] != s[m - 1]) s[m++] = s[i]; }
    *out = s;
    *nout = m;
    return 0;
}
