/*
 * CUDA backend: direct interpolation, cuFFT batched inverse transforms and the
 * exact boxcar matched filter, all on the selected device.
 *
 * The host keeps the same trial grid, chunk boundaries, analytic noise scale,
 * width banks and candidate bookkeeping as the CPU path; the device replaces
 * the three dense kernels (interpolate, inverse FFT, metric).  Candidate
 * frequencies are therefore bit-identical to the CPU path, and S/N values agree
 * to the cuFFT-vs-FFTW rounding (~1e-6 relative in Float32).
 */
#include "coherent_search.h"
#include "cuda.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

template <typename T>
struct CandRec {
    int j;
    T m;
};

static int rec_cmp_f(const void *a, const void *b)
{
    const CandRec<float> *x = (const CandRec<float> *)a;
    const CandRec<float> *y = (const CandRec<float> *)b;
    return (x->j > y->j) - (x->j < y->j);
}
static int rec_cmp_d(const void *a, const void *b)
{
    const CandRec<double> *x = (const CandRec<double> *)a;
    const CandRec<double> *y = (const CandRec<double> *)b;
    return (x->j > y->j) - (x->j < y->j);
}

/* ------------------------------------------------------------------ */
/* Device arithmetic helpers (explicit rounding, matching the CPU)     */
/* ------------------------------------------------------------------ */
static __device__ __forceinline__ float cs_dadd(float a, float b) { return __fadd_rn(a, b); }
static __device__ __forceinline__ float cs_dsub(float a, float b) { return __fsub_rn(a, b); }
static __device__ __forceinline__ float cs_dmul(float a, float b) { return __fmul_rn(a, b); }
static __device__ __forceinline__ float cs_ddiv(float a, float b) { return __fdiv_rn(a, b); }
static __device__ __forceinline__ double cs_dadd(double a, double b) { return __dadd_rn(a, b); }
static __device__ __forceinline__ double cs_dsub(double a, double b) { return __dsub_rn(a, b); }
static __device__ __forceinline__ double cs_dmul(double a, double b) { return __dmul_rn(a, b); }
static __device__ __forceinline__ double cs_ddiv(double a, double b) { return __ddiv_rn(a, b); }

static __device__ __forceinline__ void store_cplx(float2 *p, float x, float y)
{
    p->x = x; p->y = y;
}

/* Zero only the FT columns that the interpolation pass does not write: the DC
 * column and the harmonics that do not fit in this chunk.  The CPU path does
 * the same (fill_chunk_profiles); this replaces a full (nharms+1)*Nprof
 * memset of the complex spectrum per chunk. */
template <typename OUT>
__global__ void zero_cols_kernel(OUT *__restrict__ ftprofs, int nh1, int n,
                                 const int *__restrict__ cols, int ncols)
{
    long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    OUT *row = ftprofs + j * (long long)nh1;
    int c;
    for (c = 0; c < ncols; c++) {
        row[cols[c]].x = 0;
        row[cols[c]].y = 0;
    }
}
static __device__ __forceinline__ void store_cplx(double2 *p, float x, float y)
{
    p->x = (double)x; p->y = (double)y;
}

/* ------------------------------------------------------------------ */
/* Interpolation kernel                                               */
/* ------------------------------------------------------------------ */
struct ActiveEntry {
    int h;
    int res0;
    long long qint0;
};

template <typename OUT>
__global__ void interp_kernel(OUT *__restrict__ ftprofs,
                              const float2 *__restrict__ amps,
                              long long namps, int nh,
                              long long t0, int n,
                              const GpuPlan *__restrict__ plans,
                              const ActiveEntry *__restrict__ entries,
                              long long g0, int ngroups)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const long long gi = (long long)blockIdx.x * (blockDim.x >> 5) + warp;
    if (gi >= ngroups) return;
    const int aih = blockIdx.y;
    const ActiveEntry E = entries[aih];
    const int h = E.h;
    const GpuPlan P = plans[h - 1];

    const unsigned long long U =
        (unsigned long long)E.res0 +
        (unsigned long long)gi * (unsigned long long)(CS_DIRECT_GROUP_V * P.s);
    const int res = (int)(U % (unsigned long long)P.q);
    const long long qint = E.qint0 +
        (long long)gi * CS_DIRECT_GROUP_V * P.base_adv +
        (long long)(U / (unsigned long long)P.q);
    const int p = P.row[res] - 1;
    const long long base = (long long)P.rfloor0 + qint + (P.carry[p] ? 1 : 0);
    /* Amplitude window start (0-based): floor(r) + 1 - m/2.  The CPU's plane
     * buffer is 1-based internally, hence the 1 rather than 2 here. */
    const long long b0 = base + 1 - P.m / 2;
    const int g = P.grow[res] - 1;
    const long long o = P.goff[g];
    const int nj = P.gnj[g];

    float ar = 0.0f, aim = 0.0f;
    { int j; for (j = 0; j < nj; j++) {
        const long long idx = b0 + j;
        float rv = 0.0f, iv = 0.0f;
        if (idx >= 0 && idx < namps) {
            float2 a = amps[idx];
            rv = a.x; iv = a.y;
        }
        const float w = P.gW[o + (long long)j * CS_DIRECT_GROUP_V + lane];
        ar = __fmaf_rn(w, rv, ar);
        aim = __fmaf_rn(w, iv, aim);
    } }
    const float gr = P.gA_re[(long long)g * CS_DIRECT_GROUP_V + lane];
    const float gi_ = P.gA_im[(long long)g * CS_DIRECT_GROUP_V + lane];
    const float xr = __fsub_rn(__fmul_rn(gr, ar), __fmul_rn(gi_, aim));
    const float xi = __fadd_rn(__fmul_rn(gr, aim), __fmul_rn(gi_, ar));

    const long long local = (g0 + gi) * CS_DIRECT_GROUP_V + lane - t0;
    if (local >= 0 && local < n)
        store_cplx(&ftprofs[local * (long long)(nh + 1) + h], xr, xi);
}

/* ------------------------------------------------------------------ */
/* Exact per-profile boxcar metric                                    */
/* ------------------------------------------------------------------ */
/* Per-decimation parameters for the fused fold+metric kernel.  nk is the
 * folded profile length (2*Hk), woff/soff index the flat width and shape
 * banks, nvalid is the number of trials below the decimated Nyquist. */
struct FoldK {
    int k, nk, nw, wmax;
    int woff, soff;
    int nvalid, slot;
    double invsigma;
};

/* Fold + boxcar metric in one pass.
 *
 * For k | nharms the c2r of the strided harmonic stack is exactly the k-fold
 * average of the full-resolution profile, so the folded prefix sums follow
 * from the base prefix sums already in shared memory:
 *     pf[m] = (1/k) * sum_{q<k} (P[m + q*nk] - P[q*nk])
 * and the cyclic tail is pf[nk+u] = pf[nk] + pf[u].  Nothing is written to
 * global memory: the decimated profile never materialises.  The scan below is
 * arithmetic-for-arithmetic the same as metric_kernel's. */
template <typename T>
__global__ void fold_metric_kernel(const T *__restrict__ profs, int nbins, int n,
                                   const FoldK *__restrict__ dk, int ndk,
                                   const int *__restrict__ widths,
                                   const double *__restrict__ shapes,
                                   int *__restrict__ counters,
                                   CandRec<T> *__restrict__ records, int rstride,
                                   double thresh)
{
    extern __shared__ __align__(16) unsigned char smem_raw[];
    T *ps = reinterpret_cast<T *>(smem_raw);            /* (nbins+1) * 32 */
    T *pf = ps + (size_t)(nbins + 1) * 32;              /* (max nk+wmax+1) * 32 */
    const int b = threadIdx.x;
    const long long j = (long long)blockIdx.x * blockDim.x + b;
    if (j >= n) return;

    const T *col = profs + j * (long long)nbins;
    T p = (T)0;
    ps[b] = (T)0;
    { int i; for (i = 1; i <= nbins; i++) {
        p = cs_dadd(p, col[i - 1]);
        ps[(long long)i * 32 + b] = p;
    } }

    { int t; for (t = 0; t < ndk; t++) {
        const FoldK K = dk[t];
        if ((long long)K.nvalid <= j) continue;
        const int k = K.k, nk = K.nk;
        const T invk = (T)(1.0 / (double)k);
        T sc = (T)0;
        { int q; for (q = 0; q < k; q++)
            sc = cs_dadd(sc, ps[(long long)(q * nk) * 32 + b]); }
        pf[b] = (T)0;
        { int m; for (m = 1; m <= nk; m++) {
            T s = (T)0;
            { int q; for (q = 0; q < k; q++)
                s = cs_dadd(s, ps[(long long)(m + q * nk) * 32 + b]); }
            pf[(long long)m * 32 + b] = cs_dmul(cs_dsub(s, sc), invk);
        } }
        const T tot = pf[(long long)nk * 32 + b];
        { int u; for (u = 1; u <= K.wmax; u++)
            pf[(long long)(nk + u) * 32 + b] =
                cs_dadd(tot, pf[(long long)u * 32 + b]); }
        T best = (T)(-INFINITY);
        { int i; for (i = 0; i < K.nw; i++) {
            const int w = widths[K.woff + i];
            const T duty = cs_ddiv((T)w, (T)nk);
            const T invsw = (T)(K.invsigma * shapes[K.soff + i]);
            T m0 = cs_dsub(pf[(long long)w * 32 + b], pf[b]);
            { int pp; for (pp = 2; pp <= nk; pp++) {
                T d = cs_dsub(pf[(long long)(pp + w - 1) * 32 + b],
                              pf[(long long)(pp - 1) * 32 + b]);
                if (d > m0) m0 = d;
            } }
            T cand = cs_dmul(cs_dsub(m0, cs_dmul(duty, tot)), invsw);
            if (cand > best) best = cand;
        } }
        if ((double)best > thresh) {
            int pos = atomicAdd(&counters[K.slot], 1);
            if (pos < rstride) {
                CandRec<T> *r = records + (long long)K.slot * rstride + pos;
                r->j = (int)j;
                r->m = best;
            }
        }
    } }
}

template <typename T>
__global__ void metric_kernel(const T *__restrict__ profs, int nbins, int n,
                              const int *__restrict__ widths, int nw, int wmax,
                              T invsigma, const double *__restrict__ shape,
                              int *__restrict__ counter,
                              CandRec<T> *__restrict__ records, int cap,
                              double thresh)
{
    extern __shared__ __align__(16) unsigned char smem_raw[];
    T *ps = reinterpret_cast<T *>(smem_raw);
    const int b = threadIdx.x;
    const long long j = (long long)blockIdx.x * blockDim.x + b;
    if (j >= n) return;

    const T *col = profs + j * (long long)nbins;
    T p = (T)0;
    ps[b] = (T)0;
    { int i; for (i = 1; i <= nbins + wmax; i++) {
        int idx = (i > nbins ? i - nbins : i) - 1;
        p = cs_dadd(p, col[idx]);
        ps[(long long)i * 32 + b] = p;
    } }

    const int to = nbins * 32;
    T best = (T)(-INFINITY);
    { int i; for (i = 0; i < nw; i++) {
        const int w = widths[i];
        const T duty = cs_ddiv((T)w, (T)nbins);
        const T invsw = (T)(invsigma * shape[i]);
        T m = cs_dsub(ps[(long long)w * 32 + b], ps[b]);
        { int pp; for (pp = 2; pp <= nbins; pp++) {
            T d = cs_dsub(ps[(long long)(pp + w - 1) * 32 + b],
                       ps[(long long)(pp - 1) * 32 + b]);
            if (d > m) m = d;
        } }
        T prod = cs_dmul(duty, cs_dsub(ps[to + b], ps[b]));
        T cand = cs_dmul(cs_dsub(m, prod), invsw);
        if (cand > best) best = cand;
    } }
    if ((double)best > thresh) {
        int pos = atomicAdd(counter, 1);
        if (pos < cap) {
            records[pos].j = (int)j;
            records[pos].m = best;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Context                                                            */
/* ------------------------------------------------------------------ */
/* Per-stream state: two streams execute consecutive chunks so the host
 * launch latency and the candidate readback of one chunk overlap the other
 * chunk's kernels.  Everything a chunk writes (spectrum, profiles, records,
 * counts, upload staging) is double-buffered per stream; read-only tables
 * (amplitudes, widths, plan arena) are shared. */
#define CS_GPU_NST 3
struct GpuDecim {
    int k, Hk, nbins, nw, wmax;
    int shapeH[CS_GPU_NST];
    int fold;      /* 1: metrics come from fold_metric_kernel (no cuFFT) */
    int slot;      /* count/record slot (decim index + 1) */
    int woff;      /* offset into the flat fold width bank */
    int soff;      /* offset into the flat fold shape bank */
    int *d_widths;
    int *h_widths;
    double *d_shape[CS_GPU_NST];
    void *d_dprofs[CS_GPU_NST];
    cufftHandle plan[CS_GPU_NST];
};

struct GpuCtx {
    int device;
    int is_f64;
    int nh, Nprof, nbins, ndecim;
    cudaStream_t st[CS_GPU_NST];
    cudaEvent_t ev[CS_GPU_NST];
    void *d_ftprofs[CS_GPU_NST];
    void *d_profs[CS_GPU_NST];
    void *d_amps;
    long long namps;

    GpuPlan *h_plans;
    GpuPlan *d_plans;
    void *d_plans_arena;
    ActiveEntry *h_entries[CS_GPU_NST];
    ActiveEntry *d_entries[CS_GPU_NST];

    cufftHandle plan_base[CS_GPU_NST];   /* one plan per stream: a cuFFT
                                           * plan's work area is shared state,
                                           * so concurrent executions of one
                                           * plan on different streams would
                                           * race inside it */
    double *d_shape_base[CS_GPU_NST];
    int *d_widths_base;
    int *h_widths_base;
    int nw_base, wmax_base;
    int base_shapeH[CS_GPU_NST];
    double *h_shape_base;   /* per-stream staging lives in h_shapes[s] */

    GpuDecim *decims;

    int *d_counts[CS_GPU_NST];   /* one per slot: base + each decimation */
    int *h_counts[CS_GPU_NST];   /* pinned */
    void *d_records[CS_GPU_NST]; /* mapped device view of h_records */
    void *h_records[CS_GPU_NST]; /* pinned, mapped: (1+ndecim) * Nprof * rec */
    size_t rec_size;

    uint8_t *h_filled;
    uint8_t *h_filled_prev;     /* last region's filled pattern */
    int region_valid;           /* h_filled_prev holds a region */
    double sig_base_r;          /* analytic sigmas for the current region */
    int nfd_base_r;
    double *sig_decim_r;
    int *nfd_decim_r;
    int *h_zerocols[CS_GPU_NST]; /* pinned staging: FT columns to zero */
    int *d_zerocols[CS_GPU_NST];
    double *h_shapes[CS_GPU_NST]; /* shape staging, one per stream */
    int max_nw;

    /* fused fold+metric path (f32 only; skipped when shared memory is short) */
    int nfold;
    FoldK *h_fold[CS_GPU_NST];       /* per-chunk work list, ndec entries */
    FoldK *d_fold[CS_GPU_NST];
    double *h_fshape[CS_GPU_NST];    /* packed shapes for the fold decims */
    double *d_fshape[CS_GPU_NST];
    int fshape_gen[CS_GPU_NST];      /* generation of d_fshape[s] contents */
    int *h_foldw;        /* packed width banks for the fold decimations */
    int *d_foldw;
    int fshape_n;
    size_t fold_shared;  /* dynamic shared bytes for fold_metric_kernel */
};

struct GpuSession {
    GpuCtx ctx;
    const SearchParams *p;
    int device;
    int Nprof;
    int ndec;
    long long total, nchunks;
    long long N, namps;
    double T, r_lo, lodr, nyq;
    long long *cstarts;
    DirectPlanSet host_plans;   /* shallow copy; owned by the plan cache */
};

template <typename T>
static T *upload_array(const T *src, size_t n)
{
    if (!src || n == 0) return NULL;
    T *d = NULL;
    cudaMalloc((void **)&d, n * sizeof(T));
    cudaMemcpy(d, src, n * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

static void free_plans(GpuPlan *plans, int n)
{
    /* All plan tables live in one packed arena; the per-field device pointers
     * are interior offsets and must not be freed individually. */
    (void)plans; (void)n;
}

/* Upload every harmonic's plan tables as one packed device arena (two
 * cudaMallocs and two memcpys instead of ~11 per harmonic).  The device
 * pointers stored in the returned host-side GpuPlan array are interior
 * offsets into that arena. */
static GpuPlan *upload_plans(const DirectPlanSet *dplans, GpuPlan **out_dev,
                             void **out_arena)
{
    const int nh = dplans->n;
    GpuPlan *h = (GpuPlan *)calloc((size_t)nh, sizeof(GpuPlan));
    if (!h) return NULL;
    const size_t AL = 256;
    size_t nfields[nh][11];   /* padded field sizes */
    size_t rfields[nh][11];   /* raw field sizes (valid bytes to copy) */
    size_t total = 0;
    { int i; for (i = 0; i < nh; i++) {
        const DirectPlan *sp = &dplans->plans[i];
        size_t gw_n = (size_t)(sp->ngrp > 0 ? sp->goff[sp->ngrp] : 0);
        size_t fl[11] = {
            (size_t)sp->q * sizeof(int32_t),
            (size_t)sp->m * (size_t)sp->P * sizeof(float),
            (size_t)sp->P * sizeof(float),
            (size_t)sp->P * sizeof(float),
            (size_t)sp->P,
            (size_t)sp->q * sizeof(int32_t),
            (size_t)(sp->ngrp + 1) * sizeof(int32_t),
            (size_t)sp->ngrp * sizeof(int32_t),
            gw_n * sizeof(float),
            (size_t)CS_DIRECT_GROUP_V * (size_t)sp->ngrp * sizeof(float),
            (size_t)CS_DIRECT_GROUP_V * (size_t)sp->ngrp * sizeof(float),
        };
        { int f; for (f = 0; f < 11; f++) {
            rfields[i][f] = fl[f];
            nfields[i][f] = fl[f] ? (fl[f] + AL - 1) / AL * AL : 0;
            total += nfields[i][f];
        } }
    } }
    char *stage = total ? (char *)malloc(total) : NULL;
    void *arena = NULL;
    if (total) cudaMalloc(&arena, total);
    if (total && (!stage || !arena)) { free(stage); if (arena) cudaFree(arena); free(h); return NULL; }
    size_t off = 0;
    { int i; for (i = 0; i < nh; i++) {
        const DirectPlan *sp = &dplans->plans[i];
        GpuPlan *g = &h[i];
        const void *src[11] = {
            sp->row, sp->W, sp->A_re, sp->A_im, sp->carry, sp->grow,
            sp->goff, sp->gnj, sp->gW, sp->gA_re, sp->gA_im,
        };
        void *dst[11];
        { int f; for (f = 0; f < 11; f++) {
            dst[f] = nfields[i][f] ? (char *)arena + off : NULL;
            if (rfields[i][f]) memcpy((char *)stage + off, src[f], rfields[i][f]);
            off += nfields[i][f];
        } }
        g->h = sp->h; g->m = sp->m; g->q = sp->q; g->s = sp->s;
        g->base_adv = sp->base_adv; g->pnum = sp->pnum;
        g->rfloor0 = sp->rfloor0; g->P = sp->P;
        g->row = (int *)dst[0];
        g->W = (float *)dst[1];
        g->A_re = (float *)dst[2];
        g->A_im = (float *)dst[3];
        g->carry = (uint8_t *)dst[4];
        g->ngrp = sp->ngrp;
        g->grow = (int32_t *)dst[5];
        g->goff = (int32_t *)dst[6];
        g->gnj = (int32_t *)dst[7];
        g->gW = (float *)dst[8];
        g->gA_re = (float *)dst[9];
        g->gA_im = (float *)dst[10];
    } }
    if (total)
        cudaMemcpy(arena, stage, total, cudaMemcpyHostToDevice);
    free(stage);
    GpuPlan *d = NULL;
    cudaMalloc((void **)&d, (size_t)nh * sizeof(GpuPlan));
    cudaMemcpy(d, h, (size_t)nh * sizeof(GpuPlan),
                          cudaMemcpyHostToDevice);
    *out_dev = d;
    *out_arena = arena;
    return h;
}

/* ------------------------------------------------------------------ */
/* Availability                                                       */
/* ------------------------------------------------------------------ */
static cudaError_t select_device_if_needed(int device)
{
    int current = -1;
    cudaError_t e = cudaGetDevice(&current);
    if (e == cudaSuccess && current == device) return cudaSuccess;
    return cudaSetDevice(device);
}

extern "C" int cs_gpu_available(int device, char *errbuf, size_t errlen)
{
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver) {
        if (errbuf) snprintf(errbuf, errlen, "no CUDA device available (%s)",
                             cudaGetErrorString(e));
        return -1;
    }
    if (e != cudaSuccess) {
        if (errbuf) snprintf(errbuf, errlen, "cudaGetDeviceCount failed: %s",
                             cudaGetErrorString(e));
        return -1;
    }
    if (count <= 0) {
        if (errbuf) snprintf(errbuf, errlen, "no CUDA devices found");
        return -1;
    }
    if (device < 0 || device >= count) {
        if (errbuf) snprintf(errbuf, errlen,
                             "CUDA device %d requested but %d device(s) present",
                             device, count);
        return -1;
    }
    e = select_device_if_needed(device);
    if (e != cudaSuccess) {
        if (errbuf) snprintf(errbuf, errlen, "cannot select CUDA device %d: %s",
                             device, cudaGetErrorString(e));
        return -1;
    }
    cudaDeviceProp prop;
    e = cudaGetDeviceProperties(&prop, device);
    if (e != cudaSuccess) {
        if (errbuf) snprintf(errbuf, errlen, "cudaGetDeviceProperties failed: %s",
                             cudaGetErrorString(e));
        return -1;
    }
    if (errbuf) snprintf(errbuf, errlen, "%s (device %d, cc %d.%d)",
                         prop.name, device, prop.major, prop.minor);
    return 1;
}

/* ------------------------------------------------------------------ */
/* Metric launch                                                      */
/* ------------------------------------------------------------------ */
template <typename T>
static void launch_metric(GpuCtx *ctx, int s, int slot, int nbins, int n,
                          int *d_widths, int nw, int wmax, T invsigma,
                          double *d_shape, double thresh)
{
    if (n <= 0) return;
    const T *profs = (T *)ctx->d_profs[s];
    if (slot != 0) profs = (const T *)ctx->decims[slot - 1].d_dprofs[s];
    int *counter = ctx->d_counts[s] + slot;
    CandRec<T> *rec = (CandRec<T> *)ctx->d_records[s] +
                      (size_t)slot * ctx->Nprof;
    const size_t shmem = (size_t)(nbins + wmax + 1) * 32 * sizeof(T);
    const int grid = (n + 31) / 32;
    metric_kernel<T><<<grid, 32, shmem, ctx->st[s]>>>(
        profs, nbins, n, d_widths, nw, wmax, invsigma, d_shape,
        counter, rec, ctx->Nprof, thresh);
}

/* ------------------------------------------------------------------ */
/* Search region                                                      */
/* ------------------------------------------------------------------ */
extern "C" GpuSession *cs_gpu_session_create(
        const FFTFile *ft, const SearchParams *p,
        const DirectPlanSet *dplans, const long long *cstarts, long long total,
        int Nprof, long long nchunks, double r_lo, double lodr, int device)
{
    GpuSession *S = (GpuSession *)calloc(1, sizeof(GpuSession));
    if (!S) return NULL;
    S->p = p;
    S->device = device;
    S->Nprof = Nprof;
    S->total = total;
    S->nchunks = nchunks;
    S->N = ft->N;
    S->namps = ft->namps;
    S->T = ft->T;
    S->r_lo = r_lo;
    S->lodr = lodr;
    S->nyq = (double)ft->N / 2.0;
    S->host_plans = *dplans;
    S->cstarts = (long long *)malloc((size_t)nchunks * sizeof(long long));
    if (!S->cstarts) { free(S); return NULL; }
    memcpy(S->cstarts, cstarts, (size_t)nchunks * sizeof(long long));

    GpuCtx &ctx = S->ctx;
    ctx.device = device;
    ctx.is_f64 = (p->precision == CS_PREC_F64);
    ctx.nh = p->nharms;
    ctx.Nprof = Nprof;
    ctx.nbins = 2 * ctx.nh;
    const size_t csz = ctx.is_f64 ? sizeof(double2) : sizeof(float2);
    const size_t tsz = ctx.is_f64 ? sizeof(double) : sizeof(float);

    select_device_if_needed(device);
    /* Two blocking (not cudaStreamNonBlocking) streams execute consecutive
     * chunks; ordering against the legacy default stream keeps the one-time
     * table uploads (widths, fold banks, amps) safely ahead of every chunk. */
    { int s; for (s = 0; s < CS_GPU_NST; s++) {
        cudaStreamCreate(&ctx.st[s]);
        cudaEventCreate(&ctx.ev[s]);
        ctx.base_shapeH[s] = -1;
        ctx.fshape_gen[s] = -1;
    } }

    /* amplitudes: one full upload per file happens in cs_gpu_session_run() */
    ctx.namps = ft->namps;
    cudaMalloc(&ctx.d_amps, (size_t)ctx.namps * sizeof(float2));

    /* harmonic plans */
    ctx.h_plans = upload_plans(dplans, &ctx.d_plans, &ctx.d_plans_arena);

    /* decimation factors */
    int ndec = 0;
    { int i; for (i = 0; i < p->ndecim; i++) if (p->decimations[i] > 1) ndec++; }
    ctx.ndecim = ndec;
    ctx.decims = (GpuDecim *)calloc((size_t)(ndec ? ndec : 1), sizeof(GpuDecim));

    /* workspaces: one spectrum/profile pair per stream */
    { int s; for (s = 0; s < CS_GPU_NST; s++) {
        cudaMalloc(&ctx.d_ftprofs[s], (size_t)(ctx.nh + 1) * Nprof * csz);
        cudaMalloc(&ctx.d_profs[s], (size_t)ctx.nbins * Nprof * tsz);
    } }

    int n2 = ctx.nbins;
    int inembed = ctx.nh + 1, onembed = n2;
    { int s; for (s = 0; s < CS_GPU_NST; s++) {
        cufftPlanMany(&ctx.plan_base[s], 1, &n2, &inembed, 1, ctx.nh + 1,
                                  &onembed, 1, n2,
                                  ctx.is_f64 ? CUFFT_Z2D : CUFFT_C2R, Nprof);
        cufftSetStream(ctx.plan_base[s], ctx.st[s]);
    } }

    {
        int *wbuf = (int *)malloc(256 * sizeof(int));
        ctx.nw_base = ladder_boxcar_widths(ctx.nbins, 1, p, wbuf, 256);
        ctx.wmax_base = wbuf[ctx.nw_base - 1];
        ctx.h_widths_base = (int *)malloc((size_t)ctx.nw_base * sizeof(int));
        memcpy(ctx.h_widths_base, wbuf, (size_t)ctx.nw_base * sizeof(int));
        ctx.d_widths_base = upload_array(wbuf, (size_t)ctx.nw_base);
        free(wbuf);
    }
    { int s; for (s = 0; s < CS_GPU_NST; s++)
        cudaMalloc((void **)&ctx.d_shape_base[s],
                              (size_t)ctx.nw_base * sizeof(double)); }

    { int di = 0; { int i; for (i = 0; i < p->ndecim; i++) {
        const int k = p->decimations[i];
        if (k <= 1) continue;
        GpuDecim *d = &ctx.decims[di];
        d->k = k;
        d->Hk = ctx.nh / k;
        d->nbins = 2 * d->Hk;
        d->slot = di + 1;
        int *wbuf = (int *)malloc(256 * sizeof(int));
        d->nw = ladder_boxcar_widths(d->nbins, k, p, wbuf, 256);
        d->wmax = wbuf[d->nw - 1];
        { int s; for (s = 0; s < CS_GPU_NST; s++) d->shapeH[s] = -1; }
        d->h_widths = (int *)malloc((size_t)d->nw * sizeof(int));
        memcpy(d->h_widths, wbuf, (size_t)d->nw * sizeof(int));
        d->d_widths = upload_array(wbuf, (size_t)d->nw);
        free(wbuf);
        { int s; for (s = 0; s < CS_GPU_NST; s++) {
            cudaMalloc((void **)&d->d_shape[s],
                                  (size_t)d->nw * sizeof(double));
            cudaMalloc(&d->d_dprofs[s],
                                  (size_t)d->nbins * Nprof * tsz);
        } }
        d->fold = (ctx.nh % k == 0) && (d->nbins * k == 2 * ctx.nh);
        di++;
    } } }

    /* Decide whether the fold path fits: it needs the base prefix plus the
     * largest folded prefix in shared memory and f32 keeps us under the 48 KB
     * static limit.  If not, every decimation keeps its cuFFT plan. */
    { size_t maxnk = 0, maxw = 0;
      { int d; for (d = 0; d < ndec; d++) if (ctx.decims[d].fold) {
          if ((size_t)ctx.decims[d].nbins > maxnk) maxnk = ctx.decims[d].nbins;
          if ((size_t)ctx.decims[d].wmax > maxw) maxw = ctx.decims[d].wmax;
      } }
      ctx.fold_shared = ((size_t)(ctx.nbins + 1) + maxnk + maxw + 1) * 32 * tsz;
      if (ctx.is_f64 || ctx.fold_shared > 48u * 1024u) {
          { int d; for (d = 0; d < ndec; d++) ctx.decims[d].fold = 0; }
          ctx.fold_shared = 0;
      }
    }
    { int nw_all = 0, nf = 0;
      { int d; for (d = 0; d < ndec; d++) {
          if (!ctx.decims[d].fold) continue;
          nf++;
          nw_all += ctx.decims[d].nw;
      } }
      ctx.nfold = nf;
      ctx.fshape_n = nw_all;
      if (nf > 0) {
          { int s; for (s = 0; s < CS_GPU_NST; s++) {
              cudaMallocHost((void **)&ctx.h_fold[s], (size_t)ndec * sizeof(FoldK));
              cudaMalloc((void **)&ctx.d_fold[s], (size_t)ndec * sizeof(FoldK));
              cudaMallocHost((void **)&ctx.h_fshape[s],
                             (size_t)(nw_all ? nw_all : 1) * sizeof(double));
              cudaMalloc((void **)&ctx.d_fshape[s],
                         (size_t)(nw_all ? nw_all : 1) * sizeof(double));
          } }
          ctx.h_foldw = (int *)malloc((size_t)(nw_all ? nw_all : 1) * sizeof(int));
          cudaMalloc((void **)&ctx.d_foldw,
                     (size_t)(nw_all ? nw_all : 1) * sizeof(int));
          int woff = 0, soff = 0;
          { int d; for (d = 0; d < ndec; d++) {
              GpuDecim *dd = &ctx.decims[d];
              if (!dd->fold) continue;
              dd->woff = woff;
              dd->soff = soff;
              memcpy(ctx.h_foldw + woff, dd->h_widths,
                     (size_t)dd->nw * sizeof(int));
              woff += dd->nw;
              soff += dd->nw;
          } }
          if (nw_all > 0)
              cudaMemcpy(ctx.d_foldw, ctx.h_foldw,
                         (size_t)nw_all * sizeof(int), cudaMemcpyHostToDevice);
      }
    }

    { int d; for (d = 0; d < ndec; d++) {
        GpuDecim *dd = &ctx.decims[d];
        if (dd->fold) continue;   /* no plan, no dprofs: metrics are folded */
        /* Read the decimated harmonics directly from d_ftprofs: input stride k,
         * batch distance nharms+1.  This removes the gather kernel and its
         * device-to-device staging buffer entirely. */
        int nbn = dd->nbins;
        int inemb = dd->Hk + 1, onemb = dd->nbins;
        { int s; for (s = 0; s < CS_GPU_NST; s++) {
            cufftPlanMany(&dd->plan[s], 1, &nbn, &inemb, dd->k, ctx.nh + 1,
                                      &onemb, 1, dd->nbins,
                                      ctx.is_f64 ? CUFFT_Z2D : CUFFT_C2R, Nprof);
            cufftSetStream(dd->plan[s], ctx.st[s]);
        } }
    } }

    /* metric outputs: counts, records and upload staging are per stream */
    ctx.rec_size = ctx.is_f64 ? sizeof(CandRec<double>)
                              : sizeof(CandRec<float>);
    ctx.max_nw = ctx.nw_base;
    { int i; for (i = 0; i < ndec; i++)
        if (ctx.decims[i].nw > ctx.max_nw) ctx.max_nw = ctx.decims[i].nw; }
    { int s; for (s = 0; s < CS_GPU_NST; s++) {
        cudaMalloc((void **)&ctx.d_counts[s],
                              (size_t)(ndec + 1) * sizeof(int));
        cudaMallocHost((void **)&ctx.h_counts[s],
                              (size_t)(ndec + 1) * sizeof(int));
        /* Candidate records are tiny and sparse: the metric kernels write
         * them straight into mapped pinned host memory; one event per chunk
         * leaves counts and records ready with no extra copies. */
        cudaHostAlloc(&ctx.h_records[s],
                                 (size_t)(ndec + 1) * Nprof * ctx.rec_size,
                                 cudaHostAllocMapped);
        cudaHostGetDevicePointer(&ctx.d_records[s], ctx.h_records[s], 0);
        cudaMallocHost((void **)&ctx.h_entries[s],
                                 (size_t)ctx.nh * sizeof(ActiveEntry));
        cudaMalloc((void **)&ctx.d_entries[s],
                                 (size_t)ctx.nh * sizeof(ActiveEntry));
        cudaMallocHost((void **)&ctx.h_zerocols[s],
                                 (size_t)(ctx.nh + 1) * sizeof(int));
        cudaMalloc((void **)&ctx.d_zerocols[s],
                                 (size_t)(ctx.nh + 1) * sizeof(int));
        ctx.h_shapes[s] = (double *)malloc(
            (size_t)(ndec + 1) * ctx.max_nw * sizeof(double));
    } }
    ctx.h_filled = (uint8_t *)malloc((size_t)ctx.nh);
    ctx.h_filled_prev = (uint8_t *)malloc((size_t)ctx.nh);
    ctx.region_valid = 0;
    ctx.sig_decim_r = (double *)calloc((size_t)(ndec ? ndec : 1), sizeof(double));
    ctx.nfd_decim_r = (int *)calloc((size_t)(ndec ? ndec : 1), sizeof(int));

    /* dynamic shared memory opt-in for the largest metric kernel */
    {
        int max_nb = ctx.nbins, max_wm = ctx.wmax_base;
        { int i; for (i = 0; i < ndec; i++) {
            if (ctx.decims[i].nbins > max_nb) max_nb = ctx.decims[i].nbins;
            if (ctx.decims[i].wmax > max_wm) max_wm = ctx.decims[i].wmax;
        } }
        size_t shmem = (size_t)(max_nb + max_wm + 1) * 32 * tsz;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        if ((long long)shmem > (long long)prop.sharedMemPerBlockOptin) {
            fprintf(stderr, "Error: metric kernel needs %zu bytes of shared memory "
                            "but device %d allows %zu\n", shmem, device,
                    (size_t)prop.sharedMemPerBlockOptin);
            exit(1);
        }
        if (ctx.is_f64)
            cudaFuncSetAttribute((const void *)metric_kernel<double>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        else
            cudaFuncSetAttribute((const void *)metric_kernel<float>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    }

    S->ndec = ndec;
    return S;
}
extern "C" CandidateList cs_gpu_session_run(GpuSession *S, const FFTFile *ft,
                                             double threshold)
{
    CandidateList cands;
    candlist_init(&cands);
    if (!S) return cands;
    GpuCtx &ctx = S->ctx;
    const int ndec = S->ndec;
    const int Nprof = S->Nprof;
    const long long total = S->total;
    const long long nchunks = S->nchunks;
    const long long *cstarts = S->cstarts;
    const double r_lo = S->r_lo;
    const double lodr = S->lodr;
    const double nyq = S->nyq;
    const SearchParams *p = S->p;

    if (ft->namps != ctx.namps || ft->N != S->N) {
        fprintf(stderr, "Error: GPU session built for N=%lld/%lld amps but the "
                        "input has N=%lld/%lld\n",
                S->N, ctx.namps, ft->N, ft->namps);
        return cands;
    }
    /* One full upload of the finished CPU spectrum per file.  The blocking
     * copy also retires the host buffer before this function can return, even
     * if the trial range happens to contain no chunks. */
    if (ctx.namps > 0)
        cudaMemcpy(ctx.d_amps, ft->amps,
                   (size_t)ctx.namps * sizeof(float2),
                   cudaMemcpyHostToDevice);

    /* Per-chunk launch.  Everything a chunk touches is indexed by its stream
     * slot s = c % CS_GPU_NST, so chunk c and chunk c+1 run concurrently; the
     * host only waits for chunk c - CS_GPU_NST before reusing its slot. */
    double rst_ring[CS_GPU_NST];

    /* Read back one chunk's counts/records and push its candidates.  Uses only
     * values that survive unchanged from the launch (rstart, k, Hk), so the
     * candidate order and arithmetic match the single-stream driver. */
    
    for (long long c = 0; c < nchunks; c++) {
        const int s = (int)(c % CS_GPU_NST);
        cudaStream_t stm = ctx.st[s];

        if (c >= CS_GPU_NST) {
            /* The slot's previous chunk is complete; harvest its results. */
            cudaEventSynchronize(ctx.ev[s]);
            const long long cp = c - CS_GPU_NST;
            const int sp = (int)(cp % CS_GPU_NST);
            const double rstartp = rst_ring[sp];
            int cnts[64];
            { int sl; for (sl = 0; sl <= ndec && sl < 64; sl++) {
                cnts[sl] = ctx.h_counts[sp][sl];
                if (cnts[sl] > Nprof) cnts[sl] = Nprof;
            } }
            { int sl; for (sl = 0; sl <= ndec && sl < 64; sl++) {
                int cnt = cnts[sl];
                if (cnt <= 0) continue;
                char *base = (char *)ctx.h_records[sp] +
                             (size_t)sl * Nprof * ctx.rec_size;
                CandRec<float> *rf = (CandRec<float> *)base;
                CandRec<double> *rd = (CandRec<double> *)base;
                if (ctx.is_f64) qsort(rd, (size_t)cnt, ctx.rec_size, rec_cmp_d);
                else qsort(rf, (size_t)cnt, ctx.rec_size, rec_cmp_f);
                int kh = ctx.nh;
                double kmul = 1.0;
                if (sl != 0) {
                    kh = ctx.decims[sl - 1].Hk;
                    kmul = (double)ctx.decims[sl - 1].k;
                }
                { int q; for (q = 0; q < cnt; q++) {
                    int j = ctx.is_f64 ? rd[q].j : rf[q].j;
                    double metric = ctx.is_f64 ? rd[q].m : (double)rf[q].m;
                    if (!(metric > threshold)) continue;
                    double rfreq = rstartp + (double)j * lodr;
                    double rsel = kmul != 1.0 ? kmul * rfreq : rfreq;
                    if (sl != 0) {
                        if (!(rsel < nyq)) continue;
                    }
                    Candidate cd;
                    cd.freq = rsel / ft->T;
                    cd.metric = metric;
                    cd.r = rsel;
                    cd.nharm = kh;
                    cd.ducy = NAN;
                    candlist_push(&cands, cd);
                } }
            } }
        }

        cudaMemsetAsync(ctx.d_counts[s], 0,
                        (size_t)(ndec + 1) * sizeof(int), stm);
        const long long i0 = cstarts[c];
        const int n = (int)((c + 1 < nchunks ? cstarts[c + 1] : total) - i0);
        const double rstart = r_lo + (double)i0 * lodr;
        rst_ring[s] = rstart;

        /* Which harmonics carry data for this whole chunk. */
        int nactive = 0;
        const long long t_first = (i0 / CS_DIRECT_GROUP_V) * CS_DIRECT_GROUP_V;
        const long long t_last =
            ((i0 + n - 1) / CS_DIRECT_GROUP_V) * CS_DIRECT_GROUP_V +
            (CS_DIRECT_GROUP_V - 1);
        const long long g0 = t_first / CS_DIRECT_GROUP_V;
        const int ngroups = (int)(t_last / CS_DIRECT_GROUP_V - g0 + 1);

        { int h; for (h = 1; h <= ctx.nh; h++) {
            const DirectPlan *dp = &S->host_plans.plans[h - 1];
            int fits = cs_harmonic_fits(dp, ft, p, i0) &&
                       cs_harmonic_fits(dp, ft, p, i0 + n - 1);
            ctx.h_filled[h - 1] = (uint8_t)fits;
            if (fits) {
                int res;
                long long qint;
                cs_chunk_state(dp, t_first, &res, &qint);
                ctx.h_entries[s][nactive].h = h;
                ctx.h_entries[s][nactive].res0 = res;
                ctx.h_entries[s][nactive].qint0 = qint;
                nactive++;
            }
        } }

        /* The filled-harmonic pattern only changes at the change points the
         * chunk table already splits on, so most chunks sit in the same
         * region as the previous one.  Within a region the analytic sigmas
         * are constant, and each slot's inactive spectrum columns stay zero
         * (interp never writes them), so both the sigma recomputation and
         * the zero-columns upload+kernel can be skipped. */
        int region_changed =
            !ctx.region_valid ||
            memcmp(ctx.h_filled, ctx.h_filled_prev, (size_t)ctx.nh) != 0;
        if (region_changed) {
            memcpy(ctx.h_filled_prev, ctx.h_filled, (size_t)ctx.nh);
            ctx.region_valid = 1;
            ctx.sig_base_r = analytic_sigma(ctx.h_filled, 1, ctx.nh);
            ctx.nfd_base_r = 0;
            { int h; for (h = 1; h <= ctx.nh; h++)
                ctx.nfd_base_r += ctx.h_filled[h - 1] ? 1 : 0; }
            { int d; for (d = 0; d < ndec; d++) {
                GpuDecim *dd = &ctx.decims[d];
                ctx.sig_decim_r[d] = analytic_sigma(ctx.h_filled, dd->k, dd->Hk);
                ctx.nfd_decim_r[d] = 0;
                { int h; for (h = 1; h <= dd->Hk; h++)
                    ctx.nfd_decim_r[d] += ctx.h_filled[h * dd->k - 1] ? 1 : 0; }
            } }
        }

        /* Columns the interp pass will not touch: DC plus inactive harmonics. */
        /* NOTE: cuFFT's C2R destroys its input array (out-of-place included),
         * so the inactive/DC columns are clobbered by every exec and must be
         * re-zeroed each chunk; skipping this when the region is unchanged
         * leaks the clobbered values into the next chunk's transform. */
        {
          { int nc = 0;
          ctx.h_zerocols[s][nc++] = 0;
          { int h; for (h = 1; h <= ctx.nh; h++)
              if (!ctx.h_filled[h - 1]) ctx.h_zerocols[s][nc++] = h; }
          if (nc > 0) {
              cudaMemcpyAsync(ctx.d_zerocols[s], ctx.h_zerocols[s],
                              (size_t)nc * sizeof(int),
                              cudaMemcpyHostToDevice, stm);
              const int zb = 256;
              const int zg = (n + zb - 1) / zb;
              if (ctx.is_f64)
                  zero_cols_kernel<double2><<<zg, zb, 0, stm>>>(
                      (double2 *)ctx.d_ftprofs[s], ctx.nh + 1, n,
                      ctx.d_zerocols[s], nc);
              else
                  zero_cols_kernel<float2><<<zg, zb, 0, stm>>>(
                      (float2 *)ctx.d_ftprofs[s], ctx.nh + 1, n,
                      ctx.d_zerocols[s], nc);
          } }
        }
        if (nactive > 0) {
            cudaMemcpyAsync(ctx.d_entries[s], ctx.h_entries[s],
                            (size_t)nactive * sizeof(ActiveEntry),
                            cudaMemcpyHostToDevice, stm);
            const int warps = 4;
            dim3 grid((ngroups + warps - 1) / warps, nactive);
            if (ctx.is_f64)
                interp_kernel<double2><<<grid, 32 * warps, 0, stm>>>(
                    (double2 *)ctx.d_ftprofs[s], (const float2 *)ctx.d_amps,
                    ctx.namps, ctx.nh, i0, n, ctx.d_plans, ctx.d_entries[s],
                    g0, ngroups);
            else
                interp_kernel<float2><<<grid, 32 * warps, 0, stm>>>(
                    (float2 *)ctx.d_ftprofs[s], (const float2 *)ctx.d_amps,
                    ctx.namps, ctx.nh, i0, n, ctx.d_plans, ctx.d_entries[s],
                    g0, ngroups);
        }

        /* base fold */
        if (ctx.is_f64)
            cufftExecZ2D(ctx.plan_base[s],
                         (cufftDoubleComplex *)ctx.d_ftprofs[s],
                         (double *)ctx.d_profs[s]);
        else
            cufftExecC2R(ctx.plan_base[s],
                         (cufftComplex *)ctx.d_ftprofs[s],
                         (float *)ctx.d_profs[s]);

        double sig = ctx.sig_base_r;
        int nf = ctx.nfd_base_r;
        if (ctx.base_shapeH[s] != nf) {
            boxcar_shape(ctx.h_shapes[s], ctx.nbins, ctx.h_widths_base,
                         ctx.nw_base, ctx.h_filled, 1, ctx.nh);
            cudaMemcpyAsync(ctx.d_shape_base[s], ctx.h_shapes[s],
                            (size_t)ctx.nw_base * sizeof(double),
                            cudaMemcpyHostToDevice, stm);
            ctx.base_shapeH[s] = nf;
        }
        if (ctx.is_f64) {
            double sigp = sig, invsigma = sigp > 0.0 ? 1.0 / sigp : 0.0;
            launch_metric<double>(&ctx, s, 0, ctx.nbins, n, ctx.d_widths_base,
                                  ctx.nw_base, ctx.wmax_base, invsigma,
                                  ctx.d_shape_base[s], threshold);
        } else {
            float sigp = (float)sig, invsigma = sigp > 0.0f ? 1.0f / sigp : 0.0f;
            launch_metric<float>(&ctx, s, 0, ctx.nbins, n, ctx.d_widths_base,
                                 ctx.nw_base, ctx.wmax_base, invsigma,
                                 ctx.d_shape_base[s], threshold);
        }

        /* Decimations with k | nharms: one kernel folds each of them straight
         * out of the base profile's prefix sums, so no dprofs is written or
         * read back.  Non-divisor k still use cuFFT + metric_kernel below. */
        if (ctx.nfold > 0) {
            int nfd_tot = 0, shape_dirty = 0;
            { int d; for (d = 0; d < ndec; d++) {
                GpuDecim *dd = &ctx.decims[d];
                if (!dd->fold) continue;
                int nvalid;
                { int lo = 0, hi = n;
                  while (lo < hi) {
                      int mid = lo + ((hi - lo) >> 1);
                      if ((double)dd->k * (rstart + (double)mid * lodr) < nyq)
                          lo = mid + 1;
                      else hi = mid;
                  }
                  nvalid = lo; }
                if (nvalid <= 0) continue;
                sig = ctx.sig_decim_r[d];
                int nfd = ctx.nfd_decim_r[d];
                if (dd->shapeH[s] != nfd) {
                    boxcar_shape(ctx.h_fshape[s] + dd->soff, dd->nbins,
                                 dd->h_widths, dd->nw, ctx.h_filled,
                                 dd->k, dd->Hk);
                    dd->shapeH[s] = nfd;
                    shape_dirty = 1;
                }
                FoldK F;
                F.k = dd->k; F.nk = dd->nbins; F.nw = dd->nw; F.wmax = dd->wmax;
                F.woff = dd->woff; F.soff = dd->soff;
                F.nvalid = nvalid; F.slot = dd->slot;
                F.invsigma = sig > 0.0 ? 1.0 / sig : 0.0;
                ctx.h_fold[s][nfd_tot++] = F;
            } }
            if (nfd_tot > 0) {
                cudaMemcpyAsync(ctx.d_fold[s], ctx.h_fold[s],
                                (size_t)nfd_tot * sizeof(FoldK),
                                cudaMemcpyHostToDevice, stm);
                if (shape_dirty)
                    cudaMemcpyAsync(ctx.d_fshape[s], ctx.h_fshape[s],
                                    (size_t)ctx.fshape_n * sizeof(double),
                                    cudaMemcpyHostToDevice, stm);
                const int grid_f = (n + 31) / 32;
                if (ctx.is_f64)
                    fold_metric_kernel<double><<<grid_f, 32, ctx.fold_shared, stm>>>(
                        (const double *)ctx.d_profs[s], ctx.nbins, n,
                        ctx.d_fold[s], nfd_tot, ctx.d_foldw, ctx.d_fshape[s],
                        ctx.d_counts[s], (CandRec<double> *)ctx.d_records[s],
                        ctx.Nprof, threshold);
                else
                    fold_metric_kernel<float><<<grid_f, 32, ctx.fold_shared, stm>>>(
                        (const float *)ctx.d_profs[s], ctx.nbins, n,
                        ctx.d_fold[s], nfd_tot, ctx.d_foldw, ctx.d_fshape[s],
                        ctx.d_counts[s], (CandRec<float> *)ctx.d_records[s],
                        ctx.Nprof, threshold);
            }
        }

        /* decimated folds */
        { int d; for (d = 0; d < ndec; d++) {
            GpuDecim *dd = &ctx.decims[d];
            if (dd->fold) continue;
            int nvalid;
            { int lo = 0, hi = n;
              while (lo < hi) {
                  int mid = lo + ((hi - lo) >> 1);
                  if ((double)dd->k * (rstart + (double)mid * lodr) < nyq)
                      lo = mid + 1;
                  else hi = mid;
              }
              nvalid = lo; }
            if (nvalid <= 0) continue;

            double *shape = ctx.h_shapes[s] + (size_t)(d + 1) * ctx.max_nw;
            sig = ctx.sig_decim_r[d];
            int nfd = ctx.nfd_decim_r[d];
            if (dd->shapeH[s] != nfd) {
                boxcar_shape(shape, dd->nbins, dd->h_widths, dd->nw,
                             ctx.h_filled, dd->k, dd->Hk);
                cudaMemcpyAsync(dd->d_shape[s], shape,
                                (size_t)dd->nw * sizeof(double),
                                cudaMemcpyHostToDevice, stm);
                dd->shapeH[s] = nfd;
            }

            if (ctx.is_f64)
                cufftExecZ2D(dd->plan[s],
                             (cufftDoubleComplex *)ctx.d_ftprofs[s],
                             (double *)dd->d_dprofs[s]);
            else
                cufftExecC2R(dd->plan[s],
                             (cufftComplex *)ctx.d_ftprofs[s],
                             (float *)dd->d_dprofs[s]);

            if (ctx.is_f64) {
                double sigp = sig, invsigma = sigp > 0.0 ? 1.0 / sigp : 0.0;
                launch_metric<double>(&ctx, s, d + 1, dd->nbins, nvalid,
                                      dd->d_widths, dd->nw, dd->wmax,
                                      invsigma, dd->d_shape[s], threshold);
            } else {
                float sigp = (float)sig, invsigma = sigp > 0.0f ? 1.0f / sigp : 0.0f;
                launch_metric<float>(&ctx, s, d + 1, dd->nbins, nvalid,
                                     dd->d_widths, dd->nw, dd->wmax,
                                     invsigma, dd->d_shape[s], threshold);
            }
        } }

        /* Async count readback; the event marks the slot reusable. */
        cudaMemcpyAsync(ctx.h_counts[s], ctx.d_counts[s],
                        (size_t)(ndec + 1) * sizeof(int),
                        cudaMemcpyDeviceToHost, stm);
        cudaEventRecord(ctx.ev[s], stm);
    }

    /* Drain and harvest the last CS_GPU_NST chunks. */
    for (long long c = (nchunks > CS_GPU_NST ? nchunks - CS_GPU_NST : 0);
         c < nchunks; c++) {
        const int s = (int)(c % CS_GPU_NST);
        cudaEventSynchronize(ctx.ev[s]);
        const double rstartp = rst_ring[s];
        int cnts[64];
        { int sl; for (sl = 0; sl <= ndec && sl < 64; sl++) {
            cnts[sl] = ctx.h_counts[s][sl];
            if (cnts[sl] > Nprof) cnts[sl] = Nprof;
        } }
        { int sl; for (sl = 0; sl <= ndec && sl < 64; sl++) {
            int cnt = cnts[sl];
            if (cnt <= 0) continue;
            char *base = (char *)ctx.h_records[s] +
                         (size_t)sl * Nprof * ctx.rec_size;
            CandRec<float> *rf = (CandRec<float> *)base;
            CandRec<double> *rd = (CandRec<double> *)base;
            if (ctx.is_f64) qsort(rd, (size_t)cnt, ctx.rec_size, rec_cmp_d);
            else qsort(rf, (size_t)cnt, ctx.rec_size, rec_cmp_f);
            int kh = ctx.nh;
            double kmul = 1.0;
            if (sl != 0) {
                kh = ctx.decims[sl - 1].Hk;
                kmul = (double)ctx.decims[sl - 1].k;
            }
            { int q; for (q = 0; q < cnt; q++) {
                int j = ctx.is_f64 ? rd[q].j : rf[q].j;
                double metric = ctx.is_f64 ? rd[q].m : (double)rf[q].m;
                if (!(metric > threshold)) continue;
                double rfreq = rstartp + (double)j * lodr;
                double rsel = kmul != 1.0 ? kmul * rfreq : rfreq;
                if (sl != 0) {
                    if (!(rsel < nyq)) continue;
                }
                Candidate cd;
                cd.freq = rsel / ft->T;
                cd.metric = metric;
                cd.r = rsel;
                cd.nharm = kh;
                cd.ducy = NAN;
                candlist_push(&cands, cd);
            } }
        } }
    }
    return cands;
}

extern "C" void cs_gpu_session_free(GpuSession *S)
{
    if (!S) return;
    GpuCtx &ctx = S->ctx;
    const int ndec = S->ndec;
    { int s; for (s = 0; s < CS_GPU_NST; s++) {
        cudaFree(ctx.d_counts[s]);
        cudaFreeHost(ctx.h_counts[s]);
        cudaFreeHost(ctx.h_records[s]);   /* also releases the mapped view */
        cudaFree(ctx.d_entries[s]);
        cudaFreeHost(ctx.h_entries[s]);
        cudaFree(ctx.d_zerocols[s]);
        cudaFreeHost(ctx.h_zerocols[s]);
        free(ctx.h_shapes[s]);
        cudaFree(ctx.d_shape_base[s]);
        cudaFree(ctx.d_ftprofs[s]);
        cudaFree(ctx.d_profs[s]);
    } }
    free(ctx.h_filled);
    free(ctx.h_filled_prev);
    free(ctx.sig_decim_r);
    free(ctx.nfd_decim_r);
    { int i; for (i = 0; i < ndec; i++) {
        if (!ctx.decims[i].fold) {
            { int s; for (s = 0; s < CS_GPU_NST; s++)
                cufftDestroy(ctx.decims[i].plan[s]); }
            { int s; for (s = 0; s < CS_GPU_NST; s++)
                cudaFree(ctx.decims[i].d_dprofs[s]); }
        }
        cudaFree(ctx.decims[i].d_widths);
        free(ctx.decims[i].h_widths);
        { int s; for (s = 0; s < CS_GPU_NST; s++)
            cudaFree(ctx.decims[i].d_shape[s]); }
    } }
    { int s; for (s = 0; s < CS_GPU_NST; s++) {
        if (ctx.h_fold[s]) cudaFreeHost(ctx.h_fold[s]);
        cudaFree(ctx.d_fold[s]);
        if (ctx.h_fshape[s]) cudaFreeHost(ctx.h_fshape[s]);
        cudaFree(ctx.d_fshape[s]);
    } }
    free(ctx.decims);
    { int s; for (s = 0; s < CS_GPU_NST; s++)
        cufftDestroy(ctx.plan_base[s]); }
    cudaFree(ctx.d_widths_base);
    free(ctx.h_widths_base);
    free_plans(ctx.h_plans, ctx.nh);
    cudaFree(ctx.d_plans_arena);
    cudaFree(ctx.d_plans);
    cudaFree(ctx.d_amps);
    if (ctx.h_foldw) free(ctx.h_foldw);
    cudaFree(ctx.d_foldw);
    { int s; for (s = 0; s < CS_GPU_NST; s++) {
        cudaEventDestroy(ctx.ev[s]);
        cudaStreamDestroy(ctx.st[s]);
    } }
    free(S->cstarts);
    free(S);
}

/* Single-file entry point kept for the existing call sites. */
extern "C" CandidateList cs_gpu_search_region(
        const FFTFile *ft, const SearchParams *p,
        const DirectPlanSet *dplans, const long long *cstarts, long long total,
        int Nprof, long long nchunks, double r_lo, double r_hi, double lodr,
        double threshold, int device)
{
    (void)r_hi;
    GpuSession *S = cs_gpu_session_create(ft, p, dplans, cstarts, total, Nprof,
                                          nchunks, r_lo, lodr, device);
    if (!S) {
        CandidateList empty;
        candlist_init(&empty);
        return empty;
    }
    CandidateList c = cs_gpu_session_run(S, ft, threshold);
    cs_gpu_session_free(S);
    return c;
}

extern "C" int cs_gpu_session_matches(const GpuSession *S, const FFTFile *ft,
                                      const SearchParams *p, int Nprof,
                                      long long total, double r_lo,
                                      double lodr, int device)
{
    if (!S) return 0;
    return S->p == p && S->device == device && S->Nprof == Nprof &&
           S->total == total && S->T == ft->T && S->N == ft->N &&
           S->namps == ft->namps && S->r_lo == r_lo && S->lodr == lodr;
}
