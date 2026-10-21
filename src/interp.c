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
    const int nh1 = p->nharms + 1;
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
            dst[2 * ((size_t)k * nh1 + h)] = xr;
            dst[2 * ((size_t)k * nh1 + h) + 1] = xi;
        } else {
            float *dst = (float *)ws->ftprofs;
            dst[2 * ((size_t)k * nh1 + h)] = (float)xr;
            dst[2 * ((size_t)k * nh1 + h) + 1] = (float)xi;
        }
    } }
    return 1;
}

static int fill_row_group(Workspace *ws, const DirectPlan *dp, const FFTFile *ft,
                          const SearchParams *p, long long t0, int n)
{
    const int h = dp->h, m = dp->m, m2 = m / 2;
    const int V = CS_DIRECT_GROUP_V;
    const int nh1 = p->nharms + 1;
    const long long namps = ft->namps;

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
    float *re = ws->re, *im = ws->im;
    long long ihead = 1 - lo1;
    if (ihead < 0) ihead = 0;
    if (ihead > nw) ihead = nw;
    long long itail = hi1 - namps;
    if (itail < 0) itail = 0;
    if (itail > nw - ihead) itail = nw - ihead;

    { long long i; for (i = 0; i < ihead; i++) { re[i] = 0.0f; im[i] = 0.0f; } }
    long long mid_end = nw - itail;
    { long long i; for (i = ihead; i < mid_end; i++) {
        long long idx = lo1 + i - 1; /* 0-based amplitude */
        re[i] = ft->amps[2 * idx];
        im[i] = ft->amps[2 * idx + 1];
    } }
    { long long i; for (i = mid_end; i < nw; i++) { re[i] = 0.0f; im[i] = 0.0f; } }

    const float *gW = dp->gW;
    const float *gAr = dp->gA_re, *gAi = dp->gA_im;
    const int32_t *grow = dp->grow, *goff = dp->goff, *gnj = dp->gnj;
    int res = res_e;
    long long qint = qint_e;
    long long T = T_first;
    float *ftf = (float *)ws->ftprofs;
    double *ftd = (double *)ws->ftprofs;

    while (T <= T_last) {
        int g = (int)grow[res] - 1;
        int pidx = (int)dp->row[res] - 1;
        long long b0 = (long long)dp->rfloor0 + qint +
                       (dp->carry[pidx] ? 1 : 0) + 2 - m2 - lo1;
        size_t o = (size_t)goff[g];
        int nj = (int)gnj[g];
        long long j0 = T - t0;

        float ar[CS_DIRECT_GROUP_V];
        float ai[CS_DIRECT_GROUP_V];
#if defined(__AVX2__) && CS_DIRECT_GROUP_V == 32
        /* Eight live 8-lane accumulators (4 ar + 4 ai): enough FMAs in flight
         * to cover the 4-cycle FMA latency, which is the knee the Julia
         * reference measured (DIRECT_GROUP_V). */
        __m256 ar0 = _mm256_setzero_ps(), ar1 = _mm256_setzero_ps();
        __m256 ar2 = _mm256_setzero_ps(), ar3 = _mm256_setzero_ps();
        __m256 ai0 = _mm256_setzero_ps(), ai1 = _mm256_setzero_ps();
        __m256 ai2 = _mm256_setzero_ps(), ai3 = _mm256_setzero_ps();
        { int j; for (j = 0; j < nj; j++) {
            __m256 rv = _mm256_set1_ps(re[b0 + j]);
            __m256 iv = _mm256_set1_ps(im[b0 + j]);
            const float *w = gW + o + (size_t)j * V;
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
#else
        memset(ar, 0, sizeof(ar));
        memset(ai, 0, sizeof(ai));
        { int j; for (j = 0; j < nj; j++) {
            float rv = re[b0 + j];
            float iv = im[b0 + j];
            const float *w = gW + o + (size_t)j * V;
            { int k; for (k = 0; k < V; k++) {
                ar[k] = fmaf(w[k], rv, ar[k]);
                ai[k] = fmaf(w[k], iv, ai[k]);
            } }
        } }
#endif

        int full = (j0 >= 0 && j0 + V <= n);
        { int k; for (k = 0; k < V; k++) {
            long long j = j0 + k;
            if (!full && (j < 0 || j >= n)) continue;
            float gr = gAr[(size_t)g * V + k];
            float gi = gAi[(size_t)g * V + k];
            float xr = gr * ar[k] - gi * ai[k];
            float xi = gr * ai[k] + gi * ar[k];
            if (ws->is_f64) {
                ftd[2 * ((size_t)j * nh1 + h)] = (double)xr;
                ftd[2 * ((size_t)j * nh1 + h) + 1] = (double)xi;
            } else {
                ftf[2 * ((size_t)j * nh1 + h)] = xr;
                ftf[2 * ((size_t)j * nh1 + h) + 1] = xi;
            }
        } }

        long long tot = (long long)res + (long long)V * dp->s;
        qint += (long long)V * dp->base_adv + tot / dp->q;
        res = (int)(tot % dp->q);
        T += V;
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

void fill_chunk_profiles(Workspace *ws, const DirectPlanSet *dplans,
                         const FFTFile *ft, const SearchParams *p,
                         double rstart, double lodr, int n, long long t0)
{
    (void)rstart; (void)lodr;
    const int nh1 = p->nharms + 1;
    /* Only the DC row and the harmonics that will NOT be filled need zeroing:
     * every filled harmonic writes all n columns, and columns >= n are never
     * read (the transforms and metrics only consume 0..n-1).  This replaces a
     * full (nharms+1)*Nprof memset per chunk. */
    if (ws->is_f64) {
        double *dst = (double *)ws->ftprofs;
        { int j; for (j = 0; j < n; j++) {
            dst[2 * ((size_t)j * nh1)] = 0.0;
            dst[2 * ((size_t)j * nh1) + 1] = 0.0;
        } }
    } else {
        float *dst = (float *)ws->ftprofs;
        { int j; for (j = 0; j < n; j++) {
            dst[2 * ((size_t)j * nh1)] = 0.0f;
            dst[2 * ((size_t)j * nh1) + 1] = 0.0f;
        } }
    }
    { int i; for (i = 0; i < dplans->n; i++) {
        const DirectPlan *dp = &dplans->plans[i];
        int fits = harmonic_fits(dp, ft, p, t0) &&
                   harmonic_fits(dp, ft, p, t0 + n - 1);
        if (!fits) {
            const int h = dp->h;
            if (ws->is_f64) {
                double *dst = (double *)ws->ftprofs;
                { int j; for (j = 0; j < n; j++) {
                    dst[2 * ((size_t)j * nh1 + h)] = 0.0;
                    dst[2 * ((size_t)j * nh1 + h) + 1] = 0.0;
                } }
            } else {
                float *dst = (float *)ws->ftprofs;
                { int j; for (j = 0; j < n; j++) {
                    dst[2 * ((size_t)j * nh1 + h)] = 0.0f;
                    dst[2 * ((size_t)j * nh1 + h) + 1] = 0.0f;
                } }
            }
        }
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
