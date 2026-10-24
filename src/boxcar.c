/* Boxcar matched-filter metric, exact median, noise scales, batched gate. */
#include "coherent_search.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#define INF_D (INFINITY)


/* ------------------------------------------------------------------ */
/* Quickselect / median                                               */
/* ------------------------------------------------------------------ */
#define DEFINE_SELECT(T, SUFFIX)                                              \
static inline void swap_##SUFFIX(T *v, int a, int b)                          \
{ T t = v[a]; v[a] = v[b]; v[b] = t; }                                        \
static void select_##SUFFIX(T *v, int lo, int hi, int k)                      \
{                                                                             \
    while (lo < hi) {                                                         \
        if (hi - lo < 16) {                                                   \
            { int i; for (i = lo + 1; i <= hi; i++) {                              \
                T x = v[i]; int j = i - 1;                                    \
                while (j >= lo && v[j] > x) { v[j + 1] = v[j]; j--; }         \
                v[j + 1] = x;                                                 \
            } }                                                                 \
            return;                                                           \
        }                                                                     \
        int mid = (lo + hi) >> 1;                                             \
        if (v[mid] < v[lo]) swap_##SUFFIX(v, mid, lo);                        \
        if (v[hi] < v[lo]) swap_##SUFFIX(v, hi, lo);                          \
        if (v[mid] < v[hi]) swap_##SUFFIX(v, mid, hi);                        \
        T pivot = v[hi];                                                      \
        int p;                                                                \
        if (hi - lo >= 256) {                                                 \
            int i = lo;                                                       \
            { int jj; for (jj = lo; jj <= hi - 1; jj++) {                           \
                T x = v[jj];                                                  \
                v[jj] = v[i]; v[i] = x;                                       \
                i += (x <= pivot);                                            \
            } }                                                                 \
            swap_##SUFFIX(v, i, hi);                                          \
            p = i;                                                            \
        } else {                                                              \
            int i = lo - 1;                                                   \
            { int jj; for (jj = lo; jj <= hi - 1; jj++) {                           \
                if (v[jj] <= pivot) { i++; swap_##SUFFIX(v, i, jj); }         \
            } }                                                                 \
            swap_##SUFFIX(v, i + 1, hi);                                      \
            p = i + 1;                                                        \
        }                                                                     \
        if (p == k) return;                                                   \
        if (p < k) lo = p + 1; else hi = p - 1;                               \
    }                                                                         \
}                                                                             \
void select_kth_##SUFFIX(T *v, int lo, int hi, int k)                         \
{ select_##SUFFIX(v, lo, hi, k); }                                            \
T median_##SUFFIX(T *v, int n)                                                \
{                                                                             \
    int half = n >> 1;                                                        \
    select_##SUFFIX(v, 0, n - 1, half);                                       \
    if (n & 1) return v[half];                                                \
    T upper = v[half];                                                        \
    T lower = v[0];                                                           \
    { int i; for (i = 1; i < half; i++) if (v[i] > lower) lower = v[i]; }            \
    return (T)0.5 * (lower + upper);                                          \
}

DEFINE_SELECT(float, f32)
DEFINE_SELECT(double, f64)

/* ------------------------------------------------------------------ */
/* Width banks                                                        */
/* ------------------------------------------------------------------ */
int boxcar_widths(int nbins, double fsp, double maxfrac, int *out, int cap)
{
    if (nbins < 2) return 0;
    int wmax = (int)floor(maxfrac * (double)nbins);
    if (wmax < 1) wmax = 1;
    if (wmax > nbins - 1) wmax = nbins - 1;
    int n = 0, w = 1;
    while (w <= wmax) {
        if (n < cap) out[n] = w;
        n++;
        int nw = (int)floor(fsp * (double)w);
        w = nw > w + 1 ? nw : w + 1;
    }
    return n < cap ? n : cap;
}

static int int_cmp(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

int ladder_boxcar_widths(int nbins, int k, const SearchParams *p, int *out, int cap)
{
    int full[256];
    int nf = boxcar_widths(nbins, p->boxcar_fsp, p->boxcar_maxfrac, full, 256);
    if (p->ndecim <= 1) {
        int n = nf < cap ? nf : cap;
        memcpy(out, full, (size_t)n * sizeof(int));
        return n;
    }
    int ks[64];
    int nk = p->ndecim < 64 ? p->ndecim : 64;
    memcpy(ks, p->decimations, (size_t)nk * sizeof(int));
    qsort(ks, (size_t)nk, sizeof(int), int_cmp);
    { int i; for (i = 0; i + 1 < nk; i++)
        if (ks[i + 1] > (CS_LADDER_WMAX / 2) * ks[i]) {
            int n = nf < cap ? nf : cap;
            memcpy(out, full, (size_t)n * sizeof(int));
            return n;
        } }
    int kmin = ks[0], kmax = ks[nk - 1];
    int n = 0;
    { int i; for (i = 0; i < nf; i++) {
        int w = full[i];
        if ((w >= 2 || k == kmin) && (w <= CS_LADDER_WMAX || k == kmax)) {
            if (n < cap) out[n] = w;
            n++;
        }
    } }
    if (n == 0) {
        n = nf < cap ? nf : cap;
        memcpy(out, full, (size_t)n * sizeof(int));
    }
    return n < cap ? n : cap;
}

/* ------------------------------------------------------------------ */
/* Noise scales                                                       */
/* ------------------------------------------------------------------ */
#define DEFINE_BLOCK_SIGMA(T, SUFFIX)                                         \
T block_sigma_##SUFFIX(const T *M, int nbins, int n, T *buf, int cap)          \
{                                                                             \
    long long N = (long long)nbins * n;                                       \
    if (N == 0) return (T)0;                                                  \
    int ns = 0;                                                               \
    if (N <= cap) {                                                           \
        memcpy(buf, M, (size_t)N * sizeof(T));                                \
        ns = (int)N;                                                          \
    } else {                                                                  \
        int ncol = (int)(cap / nbins);                                        \
        if (ncol < 1) ncol = 1;                                               \
        if (ncol > n) ncol = n;                                               \
        { int c; for (c = 0; c < ncol; c++) {                                      \
            long long j0 = ((long long)c * n) / ncol;                         \
            long long base = j0 * nbins;                                      \
            int take = nbins;                                                 \
            if (take > cap - ns) take = cap - ns;                             \
            if (take == 0) break;                                             \
            memcpy(buf + ns, M + base, (size_t)take * sizeof(T));             \
            ns += take;                                                        \
        } }                                                                     \
    }                                                                         \
    { int i; for (i = 0; i < ns; i++) buf[i] = (T)fabs((double)buf[i]); }            \
    T med = median_##SUFFIX(buf, ns);                                         \
    return (T)1.4826 * med;                                                   \
}

DEFINE_BLOCK_SIGMA(float, f32)
DEFINE_BLOCK_SIGMA(double, f64)

double analytic_sigma(const uint8_t *filled, int k, int Hk)
{
    int nlow = 0;
    { int j; for (j = 1; j <= Hk - 1; j++)
        if (filled[j * k - 1]) nlow++; }
    int nnyq = filled[Hk * k - 1] ? 1 : 0;
    if (nlow + nnyq < 1) return 0.0;
    return sqrt(2.0 * nlow + 0.5 * nnyq);
}

void boxcar_shape(double *shape, int nbins, const int *widths, int nw,
                  const uint8_t *filled, int k, int Hk)
{
    double sigma = analytic_sigma(filled, k, Hk);
    { int i; for (i = 0; i < nw; i++) {
        int w = widths[i];
        double v = 0.0;
        { int j; for (j = 1; j <= Hk - 1; j++) {
            if (!filled[j * k - 1]) continue;
            double d = cs_sinpi((double)(j * w) / (double)nbins) /
                       cs_sinpi((double)j / (double)nbins);
            v += 2.0 * d * d;
        } }
        if (filled[Hk * k - 1]) {
            double d = cs_sinpi(0.5 * (double)w);
            v += 0.5 * d * d;
        }
        shape[i] = (sigma > 0.0 && v > 0.0) ? sigma / sqrt(v) : 0.0;
    } }
}

/* ------------------------------------------------------------------ */
/* Prefix sums / scalar scan                                          */
/* ------------------------------------------------------------------ */
#define DEFINE_PSUM_SCAN(T, SUFFIX, FABS)                                     \
static inline void psum_##SUFFIX(T *psum, const T *col, int nbins, int wmax,  \
                                 T b)                                         \
{                                                                             \
    psum[0] = (T)0;                                                           \
    { int i; for (i = 1; i <= nbins + wmax; i++) {                                 \
        int idx = i > nbins ? i - nbins : i;                                  \
        psum[i] = psum[i - 1] + (col[idx - 1] - b);                           \
    } }                                                                         \
}                                                                             \
static inline T scan_##SUFFIX(const T *psum, const int *widths, int nw,       \
                              int nbins, T invsigma, const double *shape)     \
{                                                                             \
    T best = (T)(-INFINITY);                                                  \
    T stot = psum[nbins] - psum[0];                                           \
    { int i; for (i = 0; i < nw; i++) {                                            \
        int w = widths[i];                                                    \
        T duty = (T)w / (T)nbins;                                             \
        T invsw = (T)((double)invsigma * shape[i]);                           \
        T m = psum[w] - psum[0];                                              \
        { int p; for (p = 2; p <= nbins; p++) {                                    \
            T d = psum[p + w - 1] - psum[p - 1];                              \
            if (d > m) m = d;                                                 \
        } }                                                                     \
        T cand = (m - duty * stot) * invsw;                                   \
        if (cand > best) best = cand;                                         \
    } }                                                                         \
    return best;                                                              \
}

DEFINE_PSUM_SCAN(float, f32, 0)
DEFINE_PSUM_SCAN(double, f64, 0)

float profile_boxcar_f32(const float *profs, int j, float *psum,
                         const int *widths, int nw, int nbins,
                         float invsigma, const double *shape)
{
    if (!(invsigma > 0.0f)) return 0.0f;
    psum_f32(psum, profs + (size_t)j * nbins, nbins, widths[nw - 1], 0.0f);
    return scan_f32(psum, widths, nw, nbins, invsigma, shape);
}

double profile_boxcar_f64(const double *profs, int j, double *psum,
                          const int *widths, int nw, int nbins,
                          double invsigma, const double *shape)
{
    if (!(invsigma > 0.0)) return 0.0;
    psum_f64(psum, profs + (size_t)j * nbins, nbins, widths[nw - 1], 0.0);
    return scan_f64(psum, widths, nw, nbins, invsigma, shape);
}

/* ------------------------------------------------------------------ */
/* Batched gate                                                       */
/* ------------------------------------------------------------------ */
int boxcar_batch_init(BoxcarBatch *bb, int nbins, int wmax, int Nprof)
{
    memset(bb, 0, sizeof(*bb));
    bb->B = CS_BC_BATCH;
    bb->nbins = nbins;
    bb->wmax = wmax;
    bb->Nprof = Nprof;
    bb->psT = (float *)malloc((size_t)bb->B * (nbins + wmax + 1) * sizeof(float));
    bb->res = (float *)malloc((size_t)bb->B * sizeof(float));
    bb->mbuf = (float *)malloc((size_t)bb->B * sizeof(float));
    bb->mvals = (double *)malloc((size_t)Nprof * sizeof(double));
    if (!bb->psT || !bb->res || !bb->mbuf || !bb->mvals) return -1;
    return 0;
}

void boxcar_batch_free(BoxcarBatch *bb)
{
    if (!bb) return;
    free(bb->psT); free(bb->res); free(bb->mbuf); free(bb->mvals);
    memset(bb, 0, sizeof(*bb));
}

/* Fused transpose + prefix sum.
 *
 * Reads 8 profile columns once, accumulates each column's wrapped prefix sum
 * in a register, and writes the already-transposed prefix sums directly into
 * psT.  This removes the separate (B, nbins) tile entirely -- one write and one
 * read of ~61 KB per batch -- while producing bit-identical prefix sums to
 * `psum[i] = psum[i-1] + (col[idx] - 0)`.
 *
 * The lo/hi parameters restrict the fill to profile columns [lo, hi); the
 * caller processes the batch in sub-tiles so the transposed prefix sums it is
 * about to scan stay resident in L1D for the whole tile. */
static void prefix_transpose_f32(float *psT, const float *profs, int j0,
                                 int nbins, int wmax, int blo, int bhi)
{
    const int B = CS_BC_BATCH, BJ = CS_BC_TR_BJ;
#if defined(__AVX2__)
    { int b; for (b = blo; b < bhi; b += 8)
        _mm256_storeu_ps(psT + b, _mm256_setzero_ps()); }
#else
    { int b; for (b = blo; b < bhi; b++) psT[b] = 0.0f; }
#endif
#if defined(__AVX512F__)
    /* Two 8x8 register transposes (cheap unpack/shuffle/permute2f128 moves)
     * joined with vinsertf32x8, so each prefix chain drives 16 profiles and
     * every psT store is a full 64-byte line.  The prefix adds are the same
     * adds in the same order, so psT is bit-identical to the 8-wide path. */
#define CS_TR8_INTO(p0,p1,p2,p3,p4,p5,p6,p7, idx, out)                       \
    do {                                                                     \
        __m256 _t0 = _mm256_unpacklo_ps(_mm256_loadu_ps((p0) + (idx)),       \
                                        _mm256_loadu_ps((p1) + (idx)));      \
        __m256 _t1 = _mm256_unpackhi_ps(_mm256_loadu_ps((p0) + (idx)),       \
                                        _mm256_loadu_ps((p1) + (idx)));      \
        __m256 _t2 = _mm256_unpacklo_ps(_mm256_loadu_ps((p2) + (idx)),       \
                                        _mm256_loadu_ps((p3) + (idx)));      \
        __m256 _t3 = _mm256_unpackhi_ps(_mm256_loadu_ps((p2) + (idx)),       \
                                        _mm256_loadu_ps((p3) + (idx)));      \
        __m256 _t4 = _mm256_unpacklo_ps(_mm256_loadu_ps((p4) + (idx)),       \
                                        _mm256_loadu_ps((p5) + (idx)));      \
        __m256 _t5 = _mm256_unpackhi_ps(_mm256_loadu_ps((p4) + (idx)),       \
                                        _mm256_loadu_ps((p5) + (idx)));      \
        __m256 _t6 = _mm256_unpacklo_ps(_mm256_loadu_ps((p6) + (idx)),       \
                                        _mm256_loadu_ps((p7) + (idx)));      \
        __m256 _t7 = _mm256_unpackhi_ps(_mm256_loadu_ps((p6) + (idx)),       \
                                        _mm256_loadu_ps((p7) + (idx)));      \
        __m256 _u0 = _mm256_shuffle_ps(_t0, _t2, _MM_SHUFFLE(1, 0, 1, 0));  \
        __m256 _u1 = _mm256_shuffle_ps(_t0, _t2, _MM_SHUFFLE(3, 2, 3, 2));  \
        __m256 _u2 = _mm256_shuffle_ps(_t1, _t3, _MM_SHUFFLE(1, 0, 1, 0));  \
        __m256 _u3 = _mm256_shuffle_ps(_t1, _t3, _MM_SHUFFLE(3, 2, 3, 2));  \
        __m256 _u4 = _mm256_shuffle_ps(_t4, _t6, _MM_SHUFFLE(1, 0, 1, 0));  \
        __m256 _u5 = _mm256_shuffle_ps(_t4, _t6, _MM_SHUFFLE(3, 2, 3, 2));  \
        __m256 _u6 = _mm256_shuffle_ps(_t5, _t7, _MM_SHUFFLE(1, 0, 1, 0));  \
        __m256 _u7 = _mm256_shuffle_ps(_t5, _t7, _MM_SHUFFLE(3, 2, 3, 2));  \
        (out)[0] = _mm256_permute2f128_ps(_u0, _u4, 0x20);                   \
        (out)[1] = _mm256_permute2f128_ps(_u1, _u5, 0x20);                   \
        (out)[2] = _mm256_permute2f128_ps(_u2, _u6, 0x20);                   \
        (out)[3] = _mm256_permute2f128_ps(_u3, _u7, 0x20);                   \
        (out)[4] = _mm256_permute2f128_ps(_u0, _u4, 0x31);                   \
        (out)[5] = _mm256_permute2f128_ps(_u1, _u5, 0x31);                   \
        (out)[6] = _mm256_permute2f128_ps(_u2, _u6, 0x31);                   \
        (out)[7] = _mm256_permute2f128_ps(_u3, _u7, 0x31);                   \
    } while (0)
    (void)BJ;
    memset(psT + blo, 0, (size_t)(bhi - blo) * sizeof(float));
    { int b0; for (b0 = blo; b0 < bhi; b0 += 16) {
        const float *p0 = profs + (size_t)(j0 + b0 + 0) * nbins;
        const float *p1 = profs + (size_t)(j0 + b0 + 1) * nbins;
        const float *p2 = profs + (size_t)(j0 + b0 + 2) * nbins;
        const float *p3 = profs + (size_t)(j0 + b0 + 3) * nbins;
        const float *p4 = profs + (size_t)(j0 + b0 + 4) * nbins;
        const float *p5 = profs + (size_t)(j0 + b0 + 5) * nbins;
        const float *p6 = profs + (size_t)(j0 + b0 + 6) * nbins;
        const float *p7 = profs + (size_t)(j0 + b0 + 7) * nbins;
        const float *q0 = profs + (size_t)(j0 + b0 + 8) * nbins;
        const float *q1 = profs + (size_t)(j0 + b0 + 9) * nbins;
        const float *q2 = profs + (size_t)(j0 + b0 + 10) * nbins;
        const float *q3 = profs + (size_t)(j0 + b0 + 11) * nbins;
        const float *q4 = profs + (size_t)(j0 + b0 + 12) * nbins;
        const float *q5 = profs + (size_t)(j0 + b0 + 13) * nbins;
        const float *q6 = profs + (size_t)(j0 + b0 + 14) * nbins;
        const float *q7 = profs + (size_t)(j0 + b0 + 15) * nbins;
        __m512 acc = _mm512_setzero_ps();
        int i = 1, idx = 0, seg, len;
        for (seg = 0; seg < 2; seg++) {
            len = seg ? wmax : nbins;
            while (idx + 8 <= len) {
                __m256 ca[8], cb[8];
                CS_TR8_INTO(p0,p1,p2,p3,p4,p5,p6,p7, idx, ca);
                CS_TR8_INTO(q0,q1,q2,q3,q4,q5,q6,q7, idx, cb);
                { int k; for (k = 0; k < 8; k++) {
                    __m512 v = _mm512_insertf32x8(
                        _mm512_castps256_ps512(ca[k]), cb[k], 1);
                    acc = _mm512_add_ps(acc, v);
                    _mm512_storeu_ps(psT + (size_t)(i + k) * B + b0, acc);
                } }
                idx += 8;
                i += 8;
            }
            for (; idx < len; idx++, i++) {
                __m512 v = _mm512_setr_ps(p0[idx], p1[idx], p2[idx], p3[idx],
                                          p4[idx], p5[idx], p6[idx], p7[idx],
                                          q0[idx], q1[idx], q2[idx], q3[idx],
                                          q4[idx], q5[idx], q6[idx], q7[idx]);
                acc = _mm512_add_ps(acc, v);
                _mm512_storeu_ps(psT + (size_t)i * B + b0, acc);
            }
            idx = 0;
        }
    } }
#undef CS_TR8_INTO
#elif defined(__AVX2__) && BJ == 8
    /* 8x8 register transpose: eight row loads cover eight positions for eight
     * profiles, so the inner step costs 8 loads + 24 shuffles instead of 64
     * loads.  Two profile octets run side by side so the serial prefix-add
     * chain of one octet hides under the other's loads/shuffles.  The adds are
     * the same adds in the same order, so the prefix sums are bit-identical. */
#define CS_TR8_STEP(p0,p1,p2,p3,p4,p5,p6,p7, acc, idx, ii, boff)             \
    do {                                                                     \
        __m256 _t0 = _mm256_unpacklo_ps(_mm256_loadu_ps((p0) + (idx)),       \
                                        _mm256_loadu_ps((p1) + (idx)));      \
        __m256 _t1 = _mm256_unpackhi_ps(_mm256_loadu_ps((p0) + (idx)),       \
                                        _mm256_loadu_ps((p1) + (idx)));      \
        __m256 _t2 = _mm256_unpacklo_ps(_mm256_loadu_ps((p2) + (idx)),       \
                                        _mm256_loadu_ps((p3) + (idx)));      \
        __m256 _t3 = _mm256_unpackhi_ps(_mm256_loadu_ps((p2) + (idx)),       \
                                        _mm256_loadu_ps((p3) + (idx)));      \
        __m256 _t4 = _mm256_unpacklo_ps(_mm256_loadu_ps((p4) + (idx)),       \
                                        _mm256_loadu_ps((p5) + (idx)));      \
        __m256 _t5 = _mm256_unpackhi_ps(_mm256_loadu_ps((p4) + (idx)),       \
                                        _mm256_loadu_ps((p5) + (idx)));      \
        __m256 _t6 = _mm256_unpacklo_ps(_mm256_loadu_ps((p6) + (idx)),       \
                                        _mm256_loadu_ps((p7) + (idx)));      \
        __m256 _t7 = _mm256_unpackhi_ps(_mm256_loadu_ps((p6) + (idx)),       \
                                        _mm256_loadu_ps((p7) + (idx)));      \
        __m256 _u0 = _mm256_shuffle_ps(_t0, _t2, _MM_SHUFFLE(1, 0, 1, 0));  \
        __m256 _u1 = _mm256_shuffle_ps(_t0, _t2, _MM_SHUFFLE(3, 2, 3, 2));  \
        __m256 _u2 = _mm256_shuffle_ps(_t1, _t3, _MM_SHUFFLE(1, 0, 1, 0));  \
        __m256 _u3 = _mm256_shuffle_ps(_t1, _t3, _MM_SHUFFLE(3, 2, 3, 2));  \
        __m256 _u4 = _mm256_shuffle_ps(_t4, _t6, _MM_SHUFFLE(1, 0, 1, 0));  \
        __m256 _u5 = _mm256_shuffle_ps(_t4, _t6, _MM_SHUFFLE(3, 2, 3, 2));  \
        __m256 _u6 = _mm256_shuffle_ps(_t5, _t7, _MM_SHUFFLE(1, 0, 1, 0));  \
        __m256 _u7 = _mm256_shuffle_ps(_t5, _t7, _MM_SHUFFLE(3, 2, 3, 2));  \
        __m256 _c0 = _mm256_permute2f128_ps(_u0, _u4, 0x20);                 \
        __m256 _c1 = _mm256_permute2f128_ps(_u1, _u5, 0x20);                 \
        __m256 _c2 = _mm256_permute2f128_ps(_u2, _u6, 0x20);                 \
        __m256 _c3 = _mm256_permute2f128_ps(_u3, _u7, 0x20);                 \
        __m256 _c4 = _mm256_permute2f128_ps(_u0, _u4, 0x31);                 \
        __m256 _c5 = _mm256_permute2f128_ps(_u1, _u5, 0x31);                 \
        __m256 _c6 = _mm256_permute2f128_ps(_u2, _u6, 0x31);                 \
        __m256 _c7 = _mm256_permute2f128_ps(_u3, _u7, 0x31);                 \
        acc = _mm256_add_ps(acc, _c0);                                       \
        _mm256_storeu_ps(psT + (size_t)((ii) + 0) * B + (boff), acc);        \
        acc = _mm256_add_ps(acc, _c1);                                       \
        _mm256_storeu_ps(psT + (size_t)((ii) + 1) * B + (boff), acc);        \
        acc = _mm256_add_ps(acc, _c2);                                       \
        _mm256_storeu_ps(psT + (size_t)((ii) + 2) * B + (boff), acc);        \
        acc = _mm256_add_ps(acc, _c3);                                       \
        _mm256_storeu_ps(psT + (size_t)((ii) + 3) * B + (boff), acc);        \
        acc = _mm256_add_ps(acc, _c4);                                       \
        _mm256_storeu_ps(psT + (size_t)((ii) + 4) * B + (boff), acc);        \
        acc = _mm256_add_ps(acc, _c5);                                       \
        _mm256_storeu_ps(psT + (size_t)((ii) + 5) * B + (boff), acc);        \
        acc = _mm256_add_ps(acc, _c6);                                       \
        _mm256_storeu_ps(psT + (size_t)((ii) + 6) * B + (boff), acc);        \
        acc = _mm256_add_ps(acc, _c7);                                       \
        _mm256_storeu_ps(psT + (size_t)((ii) + 7) * B + (boff), acc);        \
    } while (0)

    { int b0; for (b0 = blo; b0 < bhi; b0 += 2 * BJ) {
        const float *p0 = profs + (size_t)(j0 + b0 + 0) * nbins;
        const float *p1 = profs + (size_t)(j0 + b0 + 1) * nbins;
        const float *p2 = profs + (size_t)(j0 + b0 + 2) * nbins;
        const float *p3 = profs + (size_t)(j0 + b0 + 3) * nbins;
        const float *p4 = profs + (size_t)(j0 + b0 + 4) * nbins;
        const float *p5 = profs + (size_t)(j0 + b0 + 5) * nbins;
        const float *p6 = profs + (size_t)(j0 + b0 + 6) * nbins;
        const float *p7 = profs + (size_t)(j0 + b0 + 7) * nbins;
        const float *q0 = profs + (size_t)(j0 + b0 + 8) * nbins;
        const float *q1 = profs + (size_t)(j0 + b0 + 9) * nbins;
        const float *q2 = profs + (size_t)(j0 + b0 + 10) * nbins;
        const float *q3 = profs + (size_t)(j0 + b0 + 11) * nbins;
        const float *q4 = profs + (size_t)(j0 + b0 + 12) * nbins;
        const float *q5 = profs + (size_t)(j0 + b0 + 13) * nbins;
        const float *q6 = profs + (size_t)(j0 + b0 + 14) * nbins;
        const float *q7 = profs + (size_t)(j0 + b0 + 15) * nbins;
        __m256 acca = _mm256_setzero_ps();
        __m256 accb = _mm256_setzero_ps();
        int i = 1;             /* psT row being written */
        int idx = 0;           /* profile bin being read */
        int seg, len;
        for (seg = 0; seg < 2; seg++) {
            len = seg ? wmax : nbins;
            while (idx + 8 <= len) {
                CS_TR8_STEP(p0,p1,p2,p3,p4,p5,p6,p7, acca, idx, i, b0);
                CS_TR8_STEP(q0,q1,q2,q3,q4,q5,q6,q7, accb, idx, i, b0 + 8);
                idx += 8;
                i += 8;
            }
            for (; idx < len; idx++, i++) {
                __m256 va = _mm256_setr_ps(p0[idx], p1[idx], p2[idx], p3[idx],
                                           p4[idx], p5[idx], p6[idx], p7[idx]);
                __m256 vb = _mm256_setr_ps(q0[idx], q1[idx], q2[idx], q3[idx],
                                           q4[idx], q5[idx], q6[idx], q7[idx]);
                acca = _mm256_add_ps(acca, va);
                _mm256_storeu_ps(psT + (size_t)i * B + b0, acca);
                accb = _mm256_add_ps(accb, vb);
                _mm256_storeu_ps(psT + (size_t)i * B + b0 + 8, accb);
            }
            idx = 0;
        }
    } }
#undef CS_TR8_STEP

#else
    { int b0; for (b0 = blo; b0 < bhi; b0 += BJ) {
        const float *src[CS_BC_TR_BJ];
        { int b; for (b = 0; b < BJ; b++) src[b] = profs + (size_t)(j0 + b0 + b) * nbins; }
        float acc[CS_BC_TR_BJ];
        { int b; for (b = 0; b < BJ; b++) acc[b] = 0.f; }
        { int i; for (i = 1; i <= nbins + wmax; i++) {
            int idx = (i > nbins ? i - nbins : i) - 1;
            float *dst = psT + (size_t)i * B + b0;
            { int b; for (b = 0; b < BJ; b++) {
                acc[b] += src[b][idx];
                dst[b] = acc[b];
            } }
        } }
    } }
#endif
}
static void prefix_transpose_f64(float *psT, const double *profs, int j0,
                                 int nbins, int wmax, int blo, int bhi)
{
    const int B = CS_BC_BATCH, BJ = CS_BC_TR_BJ;
#if defined(__AVX2__)
    { int b; for (b = blo; b < bhi; b += 8)
        _mm256_storeu_ps(psT + b, _mm256_setzero_ps()); }
#else
    { int b; for (b = blo; b < bhi; b++) psT[b] = 0.0f; }
#endif
    { int b0; for (b0 = blo; b0 < bhi; b0 += BJ) {
        const double *src[CS_BC_TR_BJ];
        { int b; for (b = 0; b < BJ; b++) src[b] = profs + (size_t)(j0 + b0 + b) * nbins; }
        float acc[CS_BC_TR_BJ];
        { int b; for (b = 0; b < BJ; b++) acc[b] = 0.f; }
        { int i; for (i = 1; i <= nbins + wmax; i++) {
            int idx = (i > nbins ? i - nbins : i) - 1;
            float *dst = psT + (size_t)i * B + b0;
            { int b; for (b = 0; b < BJ; b++) {
                acc[b] += (float)src[b][idx];
                dst[b] = acc[b];
            } }
        } }
    } }
}

static void scan_batch(BoxcarBatch *bb, const int *widths, int nw, int nbins,
                       float invsigma, const double *shape, int blo, int bhi)
{
    const int B = bb->B;
    float *psT = bb->psT, *res = bb->res, *mbuf = bb->mbuf;
    int wmax = bb->wmax;
#if defined(__AVX512F__)
    /* 16 profiles per vector.  B=128 is a multiple of 16, so no tails.
     *
     * Positions are the outer loop: psT[p-1] is loaded once and reused by
     * every ladder width, so this reads (1+nw) vectors per position instead
     * of 2*nw.  Max is exact and order-free, so the result is unchanged. */
    (void)wmax;
    int to = nbins * B;
    if (nw <= 12) {
        int wo[12];
        { int i; for (i = 0; i < nw; i++) wo[i] = widths[i] * B; }
        { int b; for (b = blo; b < bhi; b += 16) {
            __m512 mx[12];
            { int i; for (i = 0; i < nw; i++) mx[i] = _mm512_set1_ps(-INFINITY); }
            { int p; for (p = 1; p <= nbins; p++) {
                int o = (p - 1) * B;
                __m512 base = _mm512_loadu_ps(psT + o + b);
                { int i; for (i = 0; i < nw; i++)
                    mx[i] = _mm512_max_ps(mx[i],
                        _mm512_sub_ps(_mm512_loadu_ps(psT + o + wo[i] + b), base)); }
            } }
            { __m512 st = _mm512_sub_ps(_mm512_loadu_ps(psT + to + b),
                                        _mm512_loadu_ps(psT + b));
              __m512 r = _mm512_set1_ps(-INFINITY);
              { int i; for (i = 0; i < nw; i++) {
                  __m512 duty = _mm512_set1_ps((float)widths[i] / (float)nbins);
                  __m512 invsw = _mm512_set1_ps((float)((double)invsigma * shape[i]));
                  r = _mm512_max_ps(r, _mm512_mul_ps(
                          _mm512_sub_ps(mx[i], _mm512_mul_ps(duty, st)), invsw));
              } }
              _mm512_storeu_ps(res + b, r); }
        } }
        return;
    }
    { int b; for (b = blo; b < bhi; b += 16)
        _mm512_storeu_ps(res + b, _mm512_set1_ps(-INFINITY)); }
    { int i; for (i = 0; i < nw; i++) {
        int w = widths[i];
        __m512 duty = _mm512_set1_ps((float)w / (float)nbins);
        __m512 invsw = _mm512_set1_ps((float)((double)invsigma * shape[i]));
        int wo = w * B;
        { int b; for (b = blo; b < bhi; b += 16) {
            __m512 x = _mm512_sub_ps(_mm512_loadu_ps(psT + wo + b),
                                     _mm512_loadu_ps(psT + b));
            _mm512_storeu_ps(mbuf + b, x);
        } }
        { int p; for (p = 2; p <= nbins; p++) {
            int o = (p - 1) * B;
            { int b; for (b = blo; b < bhi; b += 16) {
                __m512 x = _mm512_sub_ps(_mm512_loadu_ps(psT + o + wo + b),
                                         _mm512_loadu_ps(psT + o + b));
                __m512 m = _mm512_loadu_ps(mbuf + b);
                _mm512_storeu_ps(mbuf + b, _mm512_max_ps(m, x));
            } }
        } }
        { int b; for (b = blo; b < bhi; b += 16) {
            __m512 st = _mm512_sub_ps(_mm512_loadu_ps(psT + to + b),
                                      _mm512_loadu_ps(psT + b));
            __m512 c = _mm512_mul_ps(
                _mm512_sub_ps(_mm512_loadu_ps(mbuf + b),
                              _mm512_mul_ps(duty, st)), invsw);
            __m512 r = _mm512_loadu_ps(res + b);
            _mm512_storeu_ps(res + b, _mm512_max_ps(r, c));
        } }
    } }
#elif defined(__AVX2__)
    /* GCC's auto-vectorised version of these three loops is 2.4-4.8x slower
     * than this explicit form: it versions every width iteration for aliasing
     * and never keeps the max in a register across the phase loop.  B is a
     * multiple of 8, so the vector loops need no tails. */
    (void)wmax;
    { int b; for (b = blo; b < bhi; b += 8)
        _mm256_storeu_ps(res + b, _mm256_set1_ps(-INFINITY)); }
    int to = nbins * B;
    { int i; for (i = 0; i < nw; i++) {
        int w = widths[i];
        __m256 duty = _mm256_set1_ps((float)w / (float)nbins);
        __m256 invsw = _mm256_set1_ps((float)((double)invsigma * shape[i]));
        int wo = w * B;
        { int b; for (b = blo; b < bhi; b += 8) {
            __m256 x = _mm256_sub_ps(_mm256_loadu_ps(psT + wo + b),
                                     _mm256_loadu_ps(psT + b));
            _mm256_storeu_ps(mbuf + b, x);
        } }
        { int p; for (p = 2; p <= nbins; p++) {
            int o = (p - 1) * B;
            { int b; for (b = blo; b < bhi; b += 8) {
                __m256 x = _mm256_sub_ps(_mm256_loadu_ps(psT + o + wo + b),
                                         _mm256_loadu_ps(psT + o + b));
                __m256 m = _mm256_loadu_ps(mbuf + b);
                _mm256_storeu_ps(mbuf + b, _mm256_max_ps(m, x));
            } }
        } }
        { int b; for (b = blo; b < bhi; b += 8) {
            __m256 st = _mm256_sub_ps(_mm256_loadu_ps(psT + to + b),
                                      _mm256_loadu_ps(psT + b));
            __m256 c = _mm256_mul_ps(
                _mm256_sub_ps(_mm256_loadu_ps(mbuf + b),
                              _mm256_mul_ps(duty, st)), invsw);
            __m256 r = _mm256_loadu_ps(res + b);
            _mm256_storeu_ps(res + b, _mm256_max_ps(r, c));
        } }
    } }
#else
    (void)wmax;
    { int b; for (b = blo; b < bhi; b++) res[b] = -INFINITY; }
    int to = nbins * B;
    { int i; for (i = 0; i < nw; i++) {
        int w = widths[i];
        float duty = (float)w / (float)nbins;
        float invsw = (float)((double)invsigma * shape[i]);
        int wo = w * B;
        { int b; for (b = blo; b < bhi; b++) mbuf[b] = psT[wo + b] - psT[b]; }
        { int p; for (p = 2; p <= nbins; p++) {
            int o = (p - 1) * B;
            { int b; for (b = blo; b < bhi; b++) {
                float d = psT[o + wo + b] - psT[o + b];
                if (d > mbuf[b]) mbuf[b] = d;
            } }
        } }
        { int b; for (b = blo; b < bhi; b++) {
            float c = (mbuf[b] - duty * (psT[to + b] - psT[b])) * invsw;
            if (c > res[b]) res[b] = c;
        } }
    } }
#endif
}

/* Sub-tile width for the batched gate: (nbins+wmax+1) x 32 floats of psT
 * (~17 KB for the full-resolution profile) plus the 32 source columns stay
 * resident in L1D while prefix and scan run, instead of streaming the whole
 * 128-profile batch (65+ KB) through L1D for every ladder width. */
#define CS_BC_TILE 32
#define DEFINE_GATE(T, SUFFIX, PREFIX, SCALAR)                             \
static void gate_##SUFFIX(BoxcarBatch *bb, const T *profs, int n, T *psum,    \
                          const int *widths, int nw, int nbins,               \
                          T invsigma, const double *shape)                    \
{                                                                             \
    const int B = bb->B;                                                      \
    int j0 = 0;                                                               \
    while (j0 + B <= n) {                                                     \
        { int t; for (t = 0; t < B; t += CS_BC_TILE) {                        \
            PREFIX(bb->psT, profs, j0, nbins, bb->wmax, t, t + CS_BC_TILE);   \
            scan_batch(bb, widths, nw, nbins, (float)invsigma, shape,         \
                       t, t + CS_BC_TILE);                                    \
        } }                                                                   \
        { int b; for (b = 0; b < B; b++) bb->mvals[j0 + b] = (double)bb->res[b]; }   \
        j0 += B;                                                              \
    }                                                                         \
    { int j; for (j = j0; j < n; j++)                                              \
        bb->mvals[j] = (double)SCALAR(profs, j, psum, widths, nw, nbins,      \
                                      invsigma, shape); }                       \
}

DEFINE_GATE(float, f32, prefix_transpose_f32, profile_boxcar_f32)
DEFINE_GATE(double, f64, prefix_transpose_f64, profile_boxcar_f64)

void boxcar_metrics_f32(BoxcarBatch *bb, const float *profs, int n, float *psum,
                        const int *widths, int nw, int nbins,
                        float invsigma, const double *shape, double exactcut)
{
    if (exactcut == -INFINITY || !(invsigma > 0.0f)) {
        { int j; for (j = 0; j < n; j++)
            bb->mvals[j] = (double)profile_boxcar_f32(profs, j, psum, widths, nw,
                                                      nbins, invsigma, shape); }
        return;
    }
    /* The f32 batched gate is bit-identical to profile_boxcar_f32 (same
     * prefix-sum and scan expressions on the same values -- verified
     * exhaustively against the scalar path), so no post-gate re-scan is
     * needed for the candidates it lets through.  The f64 gate below keeps
     * its re-scan: it accumulates through a float psT and is approximate. */
    gate_f32(bb, profs, n, psum, widths, nw, nbins, invsigma, shape);
}

void boxcar_metrics_f64(BoxcarBatch *bb, const double *profs, int n, double *psum,
                        const int *widths, int nw, int nbins,
                        double invsigma, const double *shape, double exactcut)
{
    if (exactcut == -INFINITY || !(invsigma > 0.0)) {
        { int j; for (j = 0; j < n; j++)
            bb->mvals[j] = profile_boxcar_f64(profs, j, psum, widths, nw,
                                              nbins, invsigma, shape); }
        return;
    }
    gate_f64(bb, profs, n, psum, widths, nw, nbins, invsigma, shape);
    { int j; for (j = 0; j < n; j++) {
        if (bb->mvals[j] < exactcut) continue;
        bb->mvals[j] = profile_boxcar_f64(profs, j, psum, widths, nw,
                                          nbins, invsigma, shape);
    } }
}

/* ------------------------------------------------------------------ */
/* Best width of an isolated profile (candidate duty cycle)           */
/* ------------------------------------------------------------------ */
int boxcar_best_width(const double *prof, int nbins, double fsp, double maxfrac,
                      int *out_w, double *out_ducy)
{
    if (nbins < 1) return -1;
    int widths[256];
    int nw = boxcar_widths(nbins, fsp, maxfrac, widths, 256);
    if (nw <= 0) { *out_w = 1; *out_ducy = 1.0 / nbins; return 0; }
    int wmax = widths[nw - 1];
    double *psum = (double *)malloc((size_t)(nbins + wmax + 1) * sizeof(double));
    if (!psum) return -1;
    double sum = 0.0;
    { int i; for (i = 0; i < nbins; i++) sum += prof[i]; }
    double mean = sum / (double)nbins;
    psum[0] = 0.0;
    { int i; for (i = 1; i <= nbins + wmax; i++) {
        int idx = i > nbins ? i - nbins : i;
        psum[i] = psum[i - 1] + (prof[idx - 1] - mean);
    } }
    double stot = psum[nbins] - psum[0];

    int Hk = nbins / 2;
    uint8_t *filled = (uint8_t *)malloc((size_t)(Hk > 0 ? Hk : 1));
    if (!filled) { free(psum); return -1; }
    { int j; for (j = 0; j < Hk; j++) filled[j] = 1; }
    double *shape = (double *)malloc((size_t)nw * sizeof(double));
    if (!shape) { free(psum); free(filled); return -1; }
    if (Hk >= 1) boxcar_shape(shape, nbins, widths, nw, filled, 1, Hk);
    else { int i; for (i = 0; i < nw; i++) shape[i] = 0.0; }

    double best = -INFINITY;
    int bestw = widths[0];
    { int i; for (i = 0; i < nw; i++) {
        int w = widths[i];
        double duty = (double)w / (double)nbins;
        double invsw = shape[i];
        double m = psum[w] - psum[0];
        { int p; for (p = 2; p <= nbins; p++) {
            double d = psum[p + w - 1] - psum[p - 1];
            if (d > m) m = d;
        } }
        double cand = (m - duty * stot) * invsw;
        if (cand > best) { best = cand; bestw = w; }
    } }
    *out_w = bestw;
    *out_ducy = (double)bestw / (double)nbins;
    free(shape);
    free(filled);
    free(psum);
    return 0;
}
