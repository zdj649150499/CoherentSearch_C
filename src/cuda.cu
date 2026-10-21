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
static int gpu_dbg(void)
{
    static int v = -1;
    if (v < 0) v = getenv("CS_GPU_DEBUG") ? 1 : 0;
    return v;
}
#define GPUDBG(...) do { if (gpu_dbg()) { fprintf(stderr, "[gpu] " __VA_ARGS__); fflush(stderr); } } while (0)

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

/* Gather every k-th harmonic row into a compact contiguous stack.  One
 * launch covers every decimation factor: grid.y selects the slot. */
struct GatherArg {
    const void *src;
    void *dst;
    int k;
    int Hk;
    int block_start;
};

template <typename C>
__global__ void gather_kernel_all(const GatherArg *__restrict__ args,
                                  int ngather, int nh1, int Nprof)
{
    int sl = 0;
    while (sl + 1 < ngather && blockIdx.x >= args[sl + 1].block_start) sl++;
    const GatherArg A = args[sl];
    const C *src = (const C *)A.src;
    C *dst = (C *)A.dst;
    const long long idx =
        (long long)(blockIdx.x - A.block_start) * blockDim.x + threadIdx.x;
    const long long total = (long long)(A.Hk + 1) * Nprof;
    if (idx >= total) return;
    const int prof = (int)(idx / (A.Hk + 1));
    const int i = (int)(idx - (long long)prof * (A.Hk + 1));
    dst[idx] = src[(long long)prof * nh1 + (long long)i * A.k];
}

/* ------------------------------------------------------------------ */
/* Exact per-profile boxcar metric                                    */
/* ------------------------------------------------------------------ */
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
struct GpuDecim {
    int k, Hk, nbins, nw, wmax;
    int shapeH;
    int *d_widths;
    int *h_widths;
    double *d_shape;
    void *d_dprofs;
    void *d_gather;
    cufftHandle plan;
};

struct GpuCtx {
    int device;
    int is_f64;
    int nh, Nprof, nbins, ndecim;
    cudaStream_t stream;
    void *d_ftprofs;
    void *d_profs;
    void *d_amps;
    long long namps;

    GpuPlan *h_plans;
    GpuPlan *d_plans;
    ActiveEntry *h_entries;
    ActiveEntry *d_entries;
    GatherArg *d_gather_args;
    int max_gather_blocks;

    cufftHandle plan_base;
    double *d_shape_base;
    int *d_widths_base;
    int *h_widths_base;
    int nw_base, wmax_base;
    int base_shapeH;

    GpuDecim *decims;

    int *d_counts;       /* one per slot: base + each decimation */
    int *h_counts;       /* pinned */
    void *d_records;     /* mapped device view of h_records */
    void *h_records;     /* pinned, mapped: (1+ndecim) * Nprof * rec_size */
    size_t rec_size;

    uint8_t *h_filled;
    double *h_shapes;
    int max_nw;
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

static void cfree_field(void *p, const char *what, int i)
{
    (void)what;
    (void)i;
    if (!p) return;
    cudaFree(p);
}

static void free_plans(GpuPlan *plans, int n)
{
    if (!plans) return;
    { int i; for (i = 0; i < n; i++) {
        cfree_field(plans[i].row, "row", i);
        cfree_field(plans[i].W, "W", i);
        cfree_field(plans[i].A_re, "A_re", i);
        cfree_field(plans[i].A_im, "A_im", i);
        cfree_field(plans[i].carry, "carry", i);
        cfree_field(plans[i].grow, "grow", i);
        cfree_field(plans[i].goff, "goff", i);
        cfree_field(plans[i].gnj, "gnj", i);
        cfree_field(plans[i].gW, "gW", i);
        cfree_field(plans[i].gA_re, "gA_re", i);
        cfree_field(plans[i].gA_im, "gA_im", i);
    } }
}

static GpuPlan *upload_plans(const DirectPlanSet *dplans, GpuPlan **out_dev)
{
    const int nh = dplans->n;
    GpuPlan *h = (GpuPlan *)calloc((size_t)nh, sizeof(GpuPlan));
    if (!h) return NULL;
    { int i; for (i = 0; i < nh; i++) {
        const DirectPlan *s = &dplans->plans[i];
        GpuPlan *g = &h[i];
        g->h = s->h; g->m = s->m; g->q = s->q; g->s = s->s;
        g->base_adv = s->base_adv; g->pnum = s->pnum;
        g->rfloor0 = s->rfloor0; g->P = s->P;
        g->row = upload_array(s->row, (size_t)s->q);
        g->W = upload_array(s->W, (size_t)s->m * s->P);
        g->A_re = upload_array(s->A_re, (size_t)s->P);
        g->A_im = upload_array(s->A_im, (size_t)s->P);
        g->carry = upload_array(s->carry, (size_t)s->P);
        g->ngrp = s->ngrp;
        g->grow = upload_array(s->grow, (size_t)s->q);
        g->goff = upload_array(s->goff, (size_t)(s->ngrp + 1));
        g->gnj = upload_array(s->gnj, (size_t)s->ngrp);
        size_t gw_n = (size_t)(s->ngrp > 0 ? s->goff[s->ngrp] : 0);
        g->gW = upload_array(s->gW, gw_n);
        g->gA_re = upload_array(s->gA_re, (size_t)CS_DIRECT_GROUP_V * s->ngrp);
        g->gA_im = upload_array(s->gA_im, (size_t)CS_DIRECT_GROUP_V * s->ngrp);
    } }
    GpuPlan *d = NULL;
    cudaMalloc((void **)&d, (size_t)nh * sizeof(GpuPlan));
    cudaMemcpy(d, h, (size_t)nh * sizeof(GpuPlan),
                          cudaMemcpyHostToDevice);
    *out_dev = d;
    return h;
}

/* ------------------------------------------------------------------ */
/* Availability                                                       */
/* ------------------------------------------------------------------ */
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
    e = cudaSetDevice(device);
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
static void launch_metric(GpuCtx *ctx, int slot, int nbins, int n,
                          int *d_widths, int nw, int wmax, T invsigma,
                          double *d_shape, double thresh)
{
    if (n <= 0) return;
    const T *profs = (T *)ctx->d_profs;
    if (slot != 0) profs = (const T *)ctx->decims[slot - 1].d_dprofs;
    int *counter = ctx->d_counts + slot;
    CandRec<T> *rec = (CandRec<T> *)ctx->d_records +
                      (size_t)slot * ctx->Nprof;
    const size_t shmem = (size_t)(nbins + wmax + 1) * 32 * sizeof(T);
    const int grid = (n + 31) / 32;
    metric_kernel<T><<<grid, 32, shmem, ctx->stream>>>(
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

    GPUDBG("enter region Nprof=%d nchunks=%lld\n", Nprof, nchunks);
    cudaSetDevice(device);
    /* A blocking (not cudaStreamNonBlocking) stream lets the per-chunk
     * device-to-host count copy act as the completion point through legacy
     * default-stream ordering, so no explicit stream synchronization is
     * needed. */
    cudaStreamCreate(&ctx.stream);
    GPUDBG("stream ok\n");

    /* amplitudes: one full upload per file happens in cs_gpu_session_run() */
    ctx.namps = ft->namps;
    cudaMalloc(&ctx.d_amps, (size_t)ctx.namps * sizeof(float2));

    /* harmonic plans */
    ctx.h_plans = upload_plans(dplans, &ctx.d_plans);
    GPUDBG("plans uploaded\n");

    /* decimation factors */
    int ndec = 0;
    { int i; for (i = 0; i < p->ndecim; i++) if (p->decimations[i] > 1) ndec++; }
    ctx.ndecim = ndec;
    ctx.decims = (GpuDecim *)calloc((size_t)(ndec ? ndec : 1), sizeof(GpuDecim));

    /* workspaces */
    cudaMalloc(&ctx.d_ftprofs,
                          (size_t)(ctx.nh + 1) * Nprof * csz);
    cudaMalloc(&ctx.d_profs, (size_t)ctx.nbins * Nprof * tsz);

    int n2 = ctx.nbins;
    int inembed = ctx.nh + 1, onembed = n2;
    cufftPlanMany(&ctx.plan_base, 1, &n2, &inembed, 1, ctx.nh + 1,
                              &onembed, 1, n2,
                              ctx.is_f64 ? CUFFT_Z2D : CUFFT_C2R, Nprof);
    cufftSetStream(ctx.plan_base, ctx.stream);

    {
        int *wbuf = (int *)malloc(256 * sizeof(int));
        ctx.nw_base = ladder_boxcar_widths(ctx.nbins, 1, p, wbuf, 256);
        ctx.wmax_base = wbuf[ctx.nw_base - 1];
    ctx.base_shapeH = -1;
        ctx.h_widths_base = (int *)malloc((size_t)ctx.nw_base * sizeof(int));
        memcpy(ctx.h_widths_base, wbuf, (size_t)ctx.nw_base * sizeof(int));
        ctx.d_widths_base = upload_array(wbuf, (size_t)ctx.nw_base);
        free(wbuf);
    }
    cudaMalloc((void **)&ctx.d_shape_base,
                          (size_t)ctx.nw_base * sizeof(double));

    int di = 0;
    { int i; for (i = 0; i < p->ndecim; i++) {
        const int k = p->decimations[i];
        if (k <= 1) continue;
        GpuDecim *d = &ctx.decims[di++];
        d->k = k;
        d->Hk = ctx.nh / k;
        d->nbins = 2 * d->Hk;
        int *wbuf = (int *)malloc(256 * sizeof(int));
        d->nw = ladder_boxcar_widths(d->nbins, k, p, wbuf, 256);
        d->wmax = wbuf[d->nw - 1];
        d->shapeH = -1;
        d->h_widths = (int *)malloc((size_t)d->nw * sizeof(int));
        memcpy(d->h_widths, wbuf, (size_t)d->nw * sizeof(int));
        d->d_widths = upload_array(wbuf, (size_t)d->nw);
        free(wbuf);
        cudaMalloc((void **)&d->d_shape,
                              (size_t)d->nw * sizeof(double));
        cudaMalloc(&d->d_dprofs,
                              (size_t)d->nbins * Nprof * tsz);
        cudaMalloc(&d->d_gather,
                              (size_t)(d->Hk + 1) * Nprof * csz);
        int nbn = d->nbins;
        int inemb = d->Hk + 1, onemb = d->nbins;
        cufftPlanMany(&d->plan, 1, &nbn, &inemb, 1, d->Hk + 1,
                                  &onemb, 1, d->nbins,
                                  ctx.is_f64 ? CUFFT_Z2D : CUFFT_C2R, Nprof);
        cufftSetStream(d->plan, ctx.stream);
    } }

    GPUDBG("decim workspaces ok\n");
    /* gather arguments: one entry per decimation factor */
    if (ndec > 0) {
        GatherArg *ga = (GatherArg *)calloc((size_t)ndec, sizeof(GatherArg));
        int block_start = 0;
        { int i; for (i = 0; i < ndec; i++) {
            GpuDecim *d = &ctx.decims[i];
            ga[i].src = ctx.d_ftprofs;
            ga[i].dst = d->d_gather;
            ga[i].k = d->k;
            ga[i].Hk = d->Hk;
            ga[i].block_start = block_start;
            long long total = (long long)(d->Hk + 1) * Nprof;
            block_start += (int)((total + 255) / 256);
        } }
        ctx.max_gather_blocks = block_start > 0 ? block_start : 1;
        cudaMalloc((void **)&ctx.d_gather_args,
                              (size_t)ndec * sizeof(GatherArg));
        cudaMemcpy(ctx.d_gather_args, ga,
                              (size_t)ndec * sizeof(GatherArg),
                              cudaMemcpyHostToDevice);
        free(ga);
    } else {
        ctx.d_gather_args = NULL;
        ctx.max_gather_blocks = 1;
    }

    /* metric outputs */
    ctx.rec_size = ctx.is_f64 ? sizeof(CandRec<double>)
                              : sizeof(CandRec<float>);
    cudaMalloc((void **)&ctx.d_counts,
                          (size_t)(ndec + 1) * sizeof(int));
    cudaMallocHost((void **)&ctx.h_counts,
                              (size_t)(ndec + 1) * sizeof(int));
    /* Candidate records are tiny and sparse: let the metric kernel write them
     * straight into mapped pinned host memory.  One stream sync per chunk then
     * leaves both counts and records ready on the host, with no second sync and
     * no device-to-host record copies. */
    cudaHostAlloc(&ctx.h_records,
                             (size_t)(ndec + 1) * Nprof * ctx.rec_size,
                             cudaHostAllocMapped);
    cudaHostGetDevicePointer(&ctx.d_records, ctx.h_records, 0);

    ctx.h_entries = (ActiveEntry *)malloc((size_t)ctx.nh * sizeof(ActiveEntry));
    cudaMalloc((void **)&ctx.d_entries,
                          (size_t)ctx.nh * sizeof(ActiveEntry));
    ctx.h_filled = (uint8_t *)malloc((size_t)ctx.nh);

    ctx.max_nw = ctx.nw_base;
    { int i; for (i = 0; i < ndec; i++)
        if (ctx.decims[i].nw > ctx.max_nw) ctx.max_nw = ctx.decims[i].nw; }
    ctx.h_shapes = (double *)malloc((size_t)(ndec + 1) * ctx.max_nw * sizeof(double));

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

    GPUDBG("allocations ok\n");
    S->ndec = ndec;
    return S;
}
extern "C" CandidateList cs_gpu_session_run(GpuSession *S, const FFTFile *ft,
                                             double threshold, int progress_mode)
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
    const size_t csz = ctx.is_f64 ? sizeof(double2) : sizeof(float2);

    if (ft->namps != ctx.namps || ft->N != S->N) {
        fprintf(stderr, "Error: GPU session built for N=%lld/%lld amps but the "
                        "input has N=%lld/%lld\n",
                S->N, ctx.namps, ft->N, ft->namps);
        return cands;
    }
    GPUDBG("run T=%.12g namps=%lld\n", ft->T, ft->namps);
    /* One full upload of the finished CPU spectrum per file.  The blocking
     * copy also retires the host buffer before this function can return, even
     * if the trial range happens to contain no chunks. */
    if (ctx.namps > 0)
        cudaMemcpy(ctx.d_amps, ft->amps,
                   (size_t)ctx.namps * sizeof(float2),
                   cudaMemcpyHostToDevice);

    { long long c; for (c = 0; c < nchunks; c++) {
        cudaMemsetAsync(ctx.d_counts, 0,
                        (size_t)(ndec + 1) * sizeof(int), ctx.stream);
        GPUDBG("chunk %lld/%lld start i0=%lld\n", c, nchunks, cstarts[c]);
        const long long i0 = cstarts[c];
        const int n = (int)((c + 1 < nchunks ? cstarts[c + 1] : total) - i0);
        const double rstart = r_lo + (double)i0 * lodr;

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
                ctx.h_entries[nactive].h = h;
                ctx.h_entries[nactive].res0 = res;
                ctx.h_entries[nactive].qint0 = qint;
                nactive++;
            }
        } }

        GPUDBG("  active=%d ngroups=%d\n", nactive, ngroups);
        cudaMemsetAsync(ctx.d_ftprofs, 0,
                        (size_t)(ctx.nh + 1) * Nprof * csz, ctx.stream);
        if (nactive > 0) {
            cudaMemcpyAsync(ctx.d_entries, ctx.h_entries,
                            (size_t)nactive * sizeof(ActiveEntry),
                            cudaMemcpyHostToDevice, ctx.stream);
            const int warps = 4;
            dim3 grid((ngroups + warps - 1) / warps, nactive);
            if (ctx.is_f64)
                interp_kernel<double2><<<grid, 32 * warps, 0, ctx.stream>>>(
                    (double2 *)ctx.d_ftprofs, (const float2 *)ctx.d_amps,
                    ctx.namps, ctx.nh, i0, n, ctx.d_plans, ctx.d_entries,
                    g0, ngroups);
            else
                interp_kernel<float2><<<grid, 32 * warps, 0, ctx.stream>>>(
                    (float2 *)ctx.d_ftprofs, (const float2 *)ctx.d_amps,
                    ctx.namps, ctx.nh, i0, n, ctx.d_plans, ctx.d_entries,
                    g0, ngroups);
        }

        GPUDBG("  interp launched\n");
        /* all decimation gathers in one launch */
        if (ndec > 0) {
            if (ctx.is_f64)
                gather_kernel_all<double2><<<ctx.max_gather_blocks, 256, 0,
                                             ctx.stream>>>(
                    ctx.d_gather_args, ndec, ctx.nh + 1, Nprof);
            else
                gather_kernel_all<float2><<<ctx.max_gather_blocks, 256, 0,
                                            ctx.stream>>>(
                    ctx.d_gather_args, ndec, ctx.nh + 1, Nprof);
        }

        /* base fold */
        if (ctx.is_f64)
            cufftExecZ2D(ctx.plan_base,
                         (cufftDoubleComplex *)ctx.d_ftprofs,
                         (double *)ctx.d_profs);
        else
            cufftExecC2R(ctx.plan_base,
                         (cufftComplex *)ctx.d_ftprofs,
                         (float *)ctx.d_profs);

        GPUDBG("  base fft ok\n");
        double sig = analytic_sigma(ctx.h_filled, 1, ctx.nh);
        int nf = 0;
        { int h; for (h = 1; h <= ctx.nh; h++) nf += ctx.h_filled[h - 1] ? 1 : 0; }
        if (ctx.base_shapeH != nf) {
            boxcar_shape(ctx.h_shapes, ctx.nbins, ctx.h_widths_base, ctx.nw_base,
                         ctx.h_filled, 1, ctx.nh);
            cudaMemcpyAsync(ctx.d_shape_base, ctx.h_shapes,
                            (size_t)ctx.nw_base * sizeof(double),
                            cudaMemcpyHostToDevice, ctx.stream);
            ctx.base_shapeH = nf;
        }
        if (ctx.is_f64) {
            double sigp = sig, invsigma = sigp > 0.0 ? 1.0 / sigp : 0.0;
            launch_metric<double>(&ctx, 0, ctx.nbins, n, ctx.d_widths_base,
                                  ctx.nw_base, ctx.wmax_base, invsigma,
                                  ctx.d_shape_base, threshold);
        } else {
            float sigp = (float)sig, invsigma = sigp > 0.0f ? 1.0f / sigp : 0.0f;
            launch_metric<float>(&ctx, 0, ctx.nbins, n, ctx.d_widths_base,
                                 ctx.nw_base, ctx.wmax_base, invsigma,
                                 ctx.d_shape_base, threshold);
        }

        /* decimated folds */
        { int d; for (d = 0; d < ndec; d++) {
            GpuDecim *dd = &ctx.decims[d];
            int nvalid = 0;
            while (nvalid < n &&
                   (double)dd->k * (rstart + (double)nvalid * lodr) < nyq)
                nvalid++;
            if (nvalid <= 0) continue;

            double *shape = ctx.h_shapes + (size_t)(d + 1) * ctx.max_nw;
            sig = analytic_sigma(ctx.h_filled, dd->k, dd->Hk);
            int nfd = 0;
            { int h; for (h = 1; h <= dd->Hk; h++)
                nfd += ctx.h_filled[h * dd->k - 1] ? 1 : 0; }
            if (dd->shapeH != nfd) {
                boxcar_shape(shape, dd->nbins, dd->h_widths, dd->nw,
                             ctx.h_filled, dd->k, dd->Hk);
                cudaMemcpyAsync(dd->d_shape, shape,
                                (size_t)dd->nw * sizeof(double),
                                cudaMemcpyHostToDevice, ctx.stream);
                dd->shapeH = nfd;
            }

            if (ctx.is_f64)
                cufftExecZ2D(dd->plan,
                             (cufftDoubleComplex *)dd->d_gather,
                             (double *)dd->d_dprofs);
            else
                cufftExecC2R(dd->plan,
                             (cufftComplex *)dd->d_gather,
                             (float *)dd->d_dprofs);

            if (ctx.is_f64) {
                double sigp = sig, invsigma = sigp > 0.0 ? 1.0 / sigp : 0.0;
                launch_metric<double>(&ctx, d + 1, dd->nbins, nvalid,
                                      dd->d_widths, dd->nw, dd->wmax,
                                      invsigma, dd->d_shape, threshold);
            } else {
                float sigp = (float)sig, invsigma = sigp > 0.0f ? 1.0f / sigp : 0.0f;
                launch_metric<float>(&ctx, d + 1, dd->nbins, nvalid,
                                     dd->d_widths, dd->nw, dd->wmax,
                                     invsigma, dd->d_shape, threshold);
            }
        } }

        GPUDBG("  decims launched, waiting for counts\n");
        /* Blocking copy on the legacy default stream: it waits for the
         * blocking compute stream, so the kernels (and their candidate-record
         * writes into mapped host memory) are complete when it returns.
         * No explicit stream-synchronization call is needed. */
        cudaMemcpy(ctx.h_counts, ctx.d_counts,
                   (size_t)(ndec + 1) * sizeof(int),
                   cudaMemcpyDeviceToHost);
        GPUDBG("  chunk %lld done, cands=%zu\n", c, cands.n);

        int cnts[64];
        { int sl; for (sl = 0; sl <= ndec && sl < 64; sl++) {
            cnts[sl] = ctx.h_counts[sl];
            if (cnts[sl] > Nprof) cnts[sl] = Nprof;
        } }

        { int sl; for (sl = 0; sl <= ndec && sl < 64; sl++) {
            int cnt = cnts[sl];
            if (cnt <= 0) continue;
            char *base = (char *)ctx.h_records +
                         (size_t)sl * Nprof * ctx.rec_size;
            if (sl == 0) {
                CandRec<float> *rf = (CandRec<float> *)base;
                CandRec<double> *rd = (CandRec<double> *)base;
                if (ctx.is_f64) qsort(rd, (size_t)cnt, ctx.rec_size, rec_cmp_d);
                else qsort(rf, (size_t)cnt, ctx.rec_size, rec_cmp_f);
                { int q; for (q = 0; q < cnt; q++) {
                    int j = ctx.is_f64 ? rd[q].j : rf[q].j;
                    double metric = ctx.is_f64 ? rd[q].m : (double)rf[q].m;
                    if (!(metric > threshold)) continue;
                    double rfreq = rstart + (double)j * lodr;
                    Candidate cd;
                    cd.freq = rfreq / ft->T;
                    cd.metric = metric;
                    cd.r = rfreq;
                    cd.nharm = ctx.nh;
                    cd.ducy = NAN;
                    candlist_push(&cands, cd);
                } }
            } else {
                int d = sl - 1;
                GpuDecim *dd = &ctx.decims[d];
                CandRec<float> *rf = (CandRec<float> *)base;
                CandRec<double> *rd = (CandRec<double> *)base;
                if (ctx.is_f64) qsort(rd, (size_t)cnt, ctx.rec_size, rec_cmp_d);
                else qsort(rf, (size_t)cnt, ctx.rec_size, rec_cmp_f);
                { int q; for (q = 0; q < cnt; q++) {
                    int j = ctx.is_f64 ? rd[q].j : rf[q].j;
                    double metric = ctx.is_f64 ? rd[q].m : (double)rf[q].m;
                    if (!(metric > threshold)) continue;
                    double r_dec = (double)dd->k *
                                   (rstart + (double)j * lodr);
                    if (!(r_dec < nyq)) continue;
                    Candidate cd;
                    cd.freq = r_dec / ft->T;
                    cd.metric = metric;
                    cd.r = r_dec;
                    cd.nharm = dd->Hk;
                    cd.ducy = NAN;
                    candlist_push(&cands, cd);
                } }
            }
        } }

        if (progress_mode != 0) {
            double frac = nchunks == 0 ? 1.0 : (double)(c + 1) / (double)nchunks;
            int pct = (int)llround(100.0 * frac);
            if (progress_mode == 2) {
                int filled = (int)llround(40.0 * frac);
                if (filled < 0) filled = 0;
                if (filled > 40) filled = 40;
                fprintf(stderr, "\r  Searching [");
                { int i; for (i = 0; i < filled; i++) fputc('#', stderr); }
                { int i; for (i = filled; i < 40; i++) fputc(' ', stderr); }
                fprintf(stderr, "] %3d%%  (%lld/%lld chunks)", pct, c + 1, nchunks);
            } else {
                fprintf(stderr, "\r  Searching: %3d%%  (%lld/%lld chunks)",
                        pct, c + 1, nchunks);
            }
            fflush(stderr);
        }
    } }
    if (progress_mode != 0) fprintf(stderr, "\n");
    return cands;
}

extern "C" void cs_gpu_session_free(GpuSession *S)
{
    if (!S) return;
    GpuCtx &ctx = S->ctx;
    const int ndec = S->ndec;
    /* cleanup: the last chunk's blocking count copy already completed all
     * stream work; cudaStreamDestroy/cudaFree handle any remaining teardown. */
    cudaFree(ctx.d_counts);
    cudaFreeHost(ctx.h_counts);
    cudaFreeHost(ctx.h_records);   /* also releases the mapped device view */
    { int i; for (i = 0; i < ndec; i++) {
        cufftDestroy(ctx.decims[i].plan);
        cudaFree(ctx.decims[i].d_widths);
        free(ctx.decims[i].h_widths);
        cudaFree(ctx.decims[i].d_shape);
        cudaFree(ctx.decims[i].d_dprofs);
        cudaFree(ctx.decims[i].d_gather);
    } }
    free(ctx.decims);
    cufftDestroy(ctx.plan_base);
    cudaFree(ctx.d_shape_base);
    cudaFree(ctx.d_widths_base);
    free(ctx.h_widths_base);
    cudaFree(ctx.d_entries);
    if (ctx.d_gather_args) cudaFree(ctx.d_gather_args);
    free(ctx.h_entries);
    free(ctx.h_filled);
    free(ctx.h_shapes);
    free_plans(ctx.h_plans, ctx.nh);
    cudaFree(ctx.d_plans);
    cudaFree(ctx.d_ftprofs);
    cudaFree(ctx.d_profs);
    cudaFree(ctx.d_amps);
    cudaStreamDestroy(ctx.stream);
    free(S->cstarts);
    free(S);
}

/* Single-file entry point kept for the existing call sites. */
extern "C" CandidateList cs_gpu_search_region(
        const FFTFile *ft, const SearchParams *p,
        const DirectPlanSet *dplans, const long long *cstarts, long long total,
        int Nprof, long long nchunks, double r_lo, double r_hi, double lodr,
        double threshold, int progress_mode, int device)
{
    (void)r_hi;
    GpuSession *S = cs_gpu_session_create(ft, p, dplans, cstarts, total, Nprof,
                                          nchunks, r_lo, lodr, device);
    if (!S) {
        CandidateList empty;
        candlist_init(&empty);
        return empty;
    }
    CandidateList c = cs_gpu_session_run(S, ft, threshold, progress_mode);
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
