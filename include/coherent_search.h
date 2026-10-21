/*
 * CoherentSearch - C port of CoherentSearch.jl
 *
 * Coherent harmonic-summing pulsar search on PRESTO-style .fft files.
 *
 * The numerical behaviour follows the Julia reference implementation exactly:
 *   - direct O(m) Fourier interpolation with tabulated Eqn.-30 weights,
 *     vectorised across DIRECT_GROUP_V consecutive trials;
 *   - one batched complex-to-real FFT of all profiles in a chunk;
 *   - the zero-mean unit-L2 boxcar matched filter, normalised with the exact
 *     band-limited noise scale (boxcar_shape);
 *   - harmonic decimation k = 1..maxdecim by strided views of the harmonic
 *     stack;
 *   - the same chunking (including forced splits where the usable harmonic
 *     set changes), candidate collapsing and output format.
 */
#ifndef COHERENT_SEARCH_H
#define COHERENT_SEARCH_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <fftw3.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Tunables (kept identical to the Julia implementation)              */
/* ------------------------------------------------------------------ */
#define CS_DIRECT_GROUP_V 32
#define CS_DIRECT_QMAX (1 << 16)
#define CS_BOXCAR_SIGMA_SAMPLES 8192
#define CS_BC_BATCH 128
#define CS_BC_TR_BJ 8
#define CS_LADDER_WMAX 6
#define CS_CPU_DEFAULT_BLOCKSIZE 2048
#define CS_SIGMA_CHECK_CHUNKS 3
#define CS_SIGMA_CHECK_TOL 0.10
#define CS_MIN_WIN_TRIALS 200

/* ------------------------------------------------------------------ */
/* PRESTO .inf / .fft                                                 */
/* ------------------------------------------------------------------ */
typedef struct {
    char *path;
    char *object;      /* may be NULL */
    int   has_epoch;
    double epoch;
    long long N;       /* number of time-series bins */
    int   has_N;
    double dt;         /* time-series bin width (s) */
    int   has_dt;
    double DM;
    int   has_DM;
} SimpleInf;

typedef struct {
    char *path;
    SimpleInf inf;
    long long N;
    double T;          /* N * dt */
    double df;         /* 1 / T */
    int dereddened;
    int detrended;
    float DC;
    float Nyquist;
    float *amps;       /* mmap'ed or in-memory: 2*namps interleaved floats */
    long long namps;
    void *map;
    size_t mapsize;
    int owns_amps;     /* free(amps) on close when set */
} FFTFile;

int  inf_read(const char *path, SimpleInf *inf);
void inf_free(SimpleInf *inf);
int  fftfile_open(FFTFile *ft, const char *path);
void fftfile_close(FFTFile *ft);

/* Zaplist (PRESTO accelsearch -zaplist). */
typedef struct {
    double freq;    /* Hz */
    double width;   /* Hz */
    int bary;       /* 1 for a 'B' (barycentric) line */
} ZapBird;

typedef struct {
    ZapBird *v;
    int n;
} ZapList;

int  zaplist_read(const char *path, ZapList *z);
void zaplist_free(ZapList *z);
int  zaplist_apply(const ZapList *z, FFTFile *ft, double baryv);

/* In-memory .dat -> realfft -> rednoise pipeline (no files written). */
typedef struct DatPipe DatPipe;
DatPipe *datpipe_create(int startwidth, int endwidth, double endfreq);
void     datpipe_free(DatPipe *p);
int      datpipe_redfft(DatPipe *p, const char *datpath, FFTFile *ft);

/* Open .fft (mmap) or .dat (internal pipeline).  Returns 0 on success and sets
 * *is_dat.  For .dat, ft->amps points into pipe-owned storage. */
int  cs_open_input(const char *path, DatPipe *pipe, FFTFile *ft, int *is_dat);

/* ------------------------------------------------------------------ */
/* Search parameters                                                  */
/* ------------------------------------------------------------------ */
typedef enum { CS_PREC_F32 = 0, CS_PREC_F64 = 1 } cs_precision;
typedef enum { CS_SIGMA_ANALYTIC = 0, CS_SIGMA_MEASURED = 1 } cs_sigma;

typedef struct {
    int nharms;
    int m;
    int numbetween;
    double hidr;
    double threshold;
    double boxcar_fsp;
    double boxcar_maxfrac;
    double boxcar_gatemargin;
    int ndecim;
    int decimations[64];
    cs_precision precision;
    cs_sigma sigma;
} SearchParams;

void params_defaults(SearchParams *p);
int  decimation_set(int nharms, int maxdecim, int *out, int outcap);

/* ------------------------------------------------------------------ */
/* Candidates                                                         */
/* ------------------------------------------------------------------ */
typedef struct {
    double freq;
    double metric;
    double r;
    int nharm;
    double ducy;
} Candidate;

typedef struct {
    Candidate *v;
    size_t n;
    size_t cap;
} CandidateList;

void candlist_init(CandidateList *l);
void candlist_free(CandidateList *l);
void candlist_push(CandidateList *l, Candidate c);
void candlist_append(CandidateList *dst, const CandidateList *src);
void candlist_sort_metric_desc(CandidateList *l);
void candlist_sort_freq(CandidateList *l);
void candlist_sort_r(CandidateList *l);
void candlist_sort_metric_desc_stable(CandidateList *l);
void candlist_sort_freq_stable(CandidateList *l);
void candlist_sort_r_stable(CandidateList *l);
CandidateList remove_duplicates(const CandidateList *in, double dr_tol);
CandidateList remove_harmonics(const CandidateList *in, int numharm, double tol);
int harmonically_related(double r1, double r2, int numharm, double tol);

/* ------------------------------------------------------------------ */
/* Direct interpolation plans                                         */
/* ------------------------------------------------------------------ */
typedef struct {
    int h;
    int m;
    int q;
    int s;
    int base_adv;
    int pnum;
    int rfloor0;
    double dr0;
    int P;
    int32_t *row;          /* q entries, 1-based row index or 0 */
    float *W;              /* (m, P), column-major: W[p*m + i] */
    float *A_re, *A_im;    /* (P,) conj(A(dr)) */
    uint8_t *carry;        /* (P,) */
    int ngrp;
    int32_t *grow;         /* q entries, 1-based group index or 0 */
    int32_t *goff;         /* ngrp+1 */
    int32_t *gnj;          /* ngrp */
    float *gW;             /* flat (V, nj) blocks */
    float *gA_re, *gA_im;  /* (V, ngrp) */
} DirectPlan;

typedef struct {
    DirectPlan *plans;
    int n;
} DirectPlanSet;

int  direct_plans_build(DirectPlanSet *set, const SearchParams *p, double r_lo);
void direct_plans_free(DirectPlanSet *set);

int  trial_grid_rational(double lodr, int *pnum, int *q);
int  harmonic_change_trials(const DirectPlanSet *set, const FFTFile *ft,
                            const SearchParams *p, long long total,
                            long long **out_bnds);
void cs_fftw_plan_lock(void);
void cs_fftw_plan_unlock(void);

void cs_chunk_state(const DirectPlan *dp, long long t, int *res, long long *qint);
int  cs_harmonic_fits(const DirectPlan *dp, const FFTFile *ft,
                      const SearchParams *p, long long t);
int  chunk_starts(const DirectPlanSet *set, const FFTFile *ft, const SearchParams *p,
                  long long total, int Nprof, long long **out, long long *nout);

/* ------------------------------------------------------------------ */
/* Boxcar matched filter                                              */
/* ------------------------------------------------------------------ */
int boxcar_widths(int nbins, double fsp, double maxfrac, int *out, int cap);
int ladder_boxcar_widths(int nbins, int k, const SearchParams *p, int *out, int cap);

void   select_kth_f32(float *v, int lo, int hi, int k);
void   select_kth_f64(double *v, int lo, int hi, int k);
float  median_f32(float *v, int n);
double median_f64(double *v, int n);

float  block_sigma_f32(const float *M, int nbins, int n, float *buf, int cap);
double block_sigma_f64(const double *M, int nbins, int n, double *buf, int cap);

double analytic_sigma(const uint8_t *filled, int k, int Hk);
void   boxcar_shape(double *shape, int nbins, const int *widths, int nw,
                    const uint8_t *filled, int k, int Hk);

typedef struct {
    int B;
    int nbins;
    int wmax;
    int Nprof;
    float *tile;   /* B*nbins */
    float *psT;    /* B*(nbins+wmax+1) */
    float *res;    /* B */
    float *mbuf;   /* B */
    double *mvals; /* Nprof */
} BoxcarBatch;

int  boxcar_batch_init(BoxcarBatch *bb, int nbins, int wmax, int Nprof);
void boxcar_batch_free(BoxcarBatch *bb);
void boxcar_gate_times(double *transpose, double *scan);

float  profile_boxcar_f32(const float *profs, int j, float *psum,
                          const int *widths, int nw, int nbins,
                          float invsigma, const double *shape);
double profile_boxcar_f64(const double *profs, int j, double *psum,
                          const int *widths, int nw, int nbins,
                          double invsigma, const double *shape);

void boxcar_metrics_f32(BoxcarBatch *bb, const float *profs, int n, float *psum,
                        const int *widths, int nw, int nbins,
                        float invsigma, const double *shape, double exactcut);
void boxcar_metrics_f64(BoxcarBatch *bb, const double *profs, int n, double *psum,
                        const int *widths, int nw, int nbins,
                        double invsigma, const double *shape, double exactcut);

/* ------------------------------------------------------------------ */
/* Metric statistics (--metricstats / --normalize)                    */
/* ------------------------------------------------------------------ */
typedef struct {
    int block, k, Hk, nbins;
    double ngoodbins, flo, fhi;
    long long n;
    double min, median, mean, std, max;
} BlockMetricStats;

typedef struct {
    int k, Hk, nbins, win;
    double flo, fhi, lo, hi, invw;
    long long *counts;
    int nb;
    long long under, over, total;
    double sum, sumsq, vmin, vmax;
} MetricHistogram;

typedef struct {
    double hist_lo, hist_hi;
    int hist_nb, nwin;
    BlockMetricStats *blocks;
    size_t nblocks, capblocks;
    MetricHistogram *hists;   /* one per k */
    size_t nhists, caphists;
    MetricHistogram *whists;  /* one per (k, window) */
    size_t nwhists, capwhists;
} MetricStats;

void metricstats_init(MetricStats *ms, double lo, double hi, int nb, int nwin);
void metricstats_free(MetricStats *ms);
void metricstats_append_block(MetricStats *ms, BlockMetricStats s);
void metricstats_append_hist(MetricStats *ms, MetricHistogram h);
void metricstats_append_whist(MetricStats *ms, MetricHistogram h);
void hist_init(MetricHistogram *h, int k, int Hk, int win, double flo, double fhi,
               double lo, double hi, int nb);
void hist_free(MetricHistogram *h);
void hist_push(MetricHistogram *h, double x);
void hist_merge(MetricHistogram *a, const MetricHistogram *b);
double hist_quantile(const MetricHistogram *h, double q);
void log_edges(double a, double b, int nwin, double *edges);
int  window_index(const double *edges, int nwin, double f);

/* ------------------------------------------------------------------ */
/* Workspaces / search                                                */
/* ------------------------------------------------------------------ */
typedef struct {
    int k, Hk, nbins, wmax;
    void *dprofs;
    fftwf_plan plan_f;
    fftw_plan  plan_d;
    int *widths;
    int nw;
    double *shape;
    int shapeH;
    void *psum;
    void *bcsig;
    int sigcap;
    BoxcarBatch batch;
    uint8_t *src; /* points at ftprofs (stride k handled by the plan) */
} DecimBuf;

typedef struct {
    int is_f64;
    SearchParams *params;
    int Nprof;
    void *ftprofs;      /* (nharms+1) * Nprof complex interleaved, P */
    void *profs;        /* 2*nharms * Nprof, P */
    fftwf_plan base_plan_f;
    fftw_plan  base_plan_d;
    int *widths;
    int nw;
    int wmax;
    double *shape;
    int shapeH;
    void *psum;
    void *bcsig;
    int sigcap;
    BoxcarBatch batch;
    DecimBuf *decims;
    int ndecim;
    float *re, *im;     /* de-interleaved amplitude planes */
    size_t planesize;
    uint8_t *filled;    /* nharms */
    void *ftprofs_base; /* same as ftprofs (for clarity) */
} Workspace;

typedef struct {
    int Nprof;
    int nthreads;
    Workspace *ws;
    SearchParams *params;
    /* File-independent search setup, reusable when the observation length and
     * trial grid are unchanged (DM sweeps, multi-file runs). */
    int plan_valid;
    double plan_T;
    double plan_lofreq, plan_hifreq;
    int plan_lobin, plan_Nprof;
    long long plan_total;
    DirectPlanSet dplans;
    long long *cstarts;
    long long nchunks;
    void *gpu_session;   /* GpuSession* when the CUDA backend is active */
    int wisdom_done;     /* FFTW wisdom imported/exported for this process */
} SearchCache;

void search_cache_free(SearchCache *cache);
int  cs_prime_wisdom(const SearchParams *p, int Nprof, int wisdom,
                     const char *wisdom_file);

Workspace *workspaces_create(const SearchParams *p, int Nprof, int nthreads,
                             unsigned plan_flags);
void       workspaces_free(Workspace *ws, int nthreads, int ndecim);

void fill_chunk_profiles(Workspace *ws, const DirectPlanSet *dplans,
                         const FFTFile *ft, const SearchParams *p,
                         double rstart, double lodr, int n, long long t0);
void transform_base(Workspace *ws);
void transform_decim(Workspace *ws, DecimBuf *db);
int  direct_window_size(const SearchParams *p, int Nprof);

typedef struct {
    int k;
    double *edges;   /* nwin+1 */
    double *loc;
    double *scale;
    int nwin;
} NormK;

typedef struct {
    NormK *ks;
    int n;
} MetricNorm;

MetricNorm *build_metricnorm(const MetricStats *ms);
void metricnorm_free(MetricNorm *n);
double metric_normalize(const MetricNorm *n, int k, double f, double M);

/* ------------------------------------------------------------------ */
/* CLI / output                                                       */
/* ------------------------------------------------------------------ */
int  cs_main(int argc, char **argv);
int  cs_main_list(int argc, char **argv);
int  write_candidates(const CandidateList *cands, const char *outfile, double threshold);
void cs_format_double(char *buf, size_t n, double x);
CandidateList measure_ducy(const FFTFile *ft, const CandidateList *in,
                           const SearchParams *p);
CandidateList read_candidates_file(const char *path, double T);
double *candidate_profile(const FFTFile *ft, double r, int nharm, int m, int *out_n);
int boxcar_best_width(const double *prof, int nbins, double fsp, double maxfrac,
                      int *out_w, double *out_ducy);
void fourier_interp_complex(double r, const float *amps, long long namps, int m,
                            double *out_re, double *out_im);

/* Search driver */
CandidateList search_file(const FFTFile *ft, const SearchParams *p,
                          double lofreq, double hifreq, int lobin, int blocksize,
                          double threshold, int remove, double dr_tol,
                          int harm_remove, int numharm, int progress_mode,
                          MetricStats *metricstats, int normalize, int verbose,
                          int wisdom, const char *wisdom_file,
                          SearchCache *cache, double harm_tol,
                          int gpu_device);

/* GPU backend.  Return 1 if a CUDA-enabled build has the requested device,
 * 0 if this build has no CUDA support, -1 on a runtime/device error. */
int cs_gpu_available(int device, char *errbuf, size_t errlen);

typedef struct GpuSession GpuSession;

GpuSession *cs_gpu_session_create(const FFTFile *ft, const SearchParams *p,
                                  const DirectPlanSet *dplans,
                                  const long long *cstarts, long long total,
                                  int Nprof, long long nchunks,
                                  double r_lo, double lodr, int device);
int cs_gpu_session_matches(const GpuSession *s, const FFTFile *ft,
                           const SearchParams *p, int Nprof, long long total,
                           double r_lo, double lodr, int device);
CandidateList cs_gpu_session_run(GpuSession *s, const FFTFile *ft,
                                 double threshold, int progress_mode);
void cs_gpu_session_free(GpuSession *s);

CandidateList cs_gpu_search_region(const FFTFile *ft, const SearchParams *p,
                                   const DirectPlanSet *dplans,
                                   const long long *cstarts, long long total,
                                   int Nprof, long long nchunks,
                                   double r_lo, double r_hi, double lodr,
                                   double threshold, int progress_mode,
                                   int device);

void report_metricstats(const MetricStats *ms, const char *base,
                        const SearchParams *p, double threshold);

int plot_candidates_svg(const FFTFile *ft, const CandidateList *cands,
                        const SearchParams *p, double threshold,
                        const char *outstem, int ncols, int nrows);

double cs_sinpi(double x);
double cs_cospi(double x);
double cs_eps(double x);

#ifdef __cplusplus
}
#endif

#endif /* COHERENT_SEARCH_H */
