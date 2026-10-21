/* Search driver: workspaces, chunk loop, decimation, statistics, normalisation. */
#include "coherent_search.h"

#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef INFINITY
#define INFINITY (1.0 / 0.0)
#endif

/* ------------------------------------------------------------------ */
/* Small helpers                                                      */
/* ------------------------------------------------------------------ */
static int int_cmp_local(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static int block_cmp(const void *a, const void *b)
{
    const BlockMetricStats *x = (const BlockMetricStats *)a;
    const BlockMetricStats *y = (const BlockMetricStats *)b;
    if (x->block != y->block) return (x->block > y->block) - (x->block < y->block);
    return (x->k > y->k) - (x->k < y->k);
}

static void refresh_shape(double *shape, int *shapeH, int nbins,
                          const int *widths, int nw,
                          const uint8_t *filled, int k, int Hk)
{
    int nf = 0;
    { int j; for (j = 1; j <= Hk; j++) if (filled[j * k - 1]) nf++; }
    if (*shapeH != nf) {
        boxcar_shape(shape, nbins, widths, nw, filled, k, Hk);
        *shapeH = nf;
    }
}

/* ------------------------------------------------------------------ */
/* Workspace                                                          */
/* ------------------------------------------------------------------ */
static int build_decim(DecimBuf *db, int k, Workspace *ws, const SearchParams *p,
                       unsigned flags)
{
    memset(db, 0, sizeof(*db));
    const int nh = p->nharms;
    const int Nprof = ws->Nprof;
    db->k = k;
    db->Hk = nh / k;
    db->nbins = 2 * db->Hk;
    db->nw = ladder_boxcar_widths(db->nbins, k, p, db->widths = (int *)malloc(256 * sizeof(int)), 256);
    if (!db->widths) return -1;
    db->wmax = db->widths[db->nw - 1];
    int n2 = db->nbins;
    if (ws->is_f64) {
        double *d = (double *)fftw_malloc((size_t)db->nbins * Nprof * sizeof(double));
        if (!d) return -1;
        db->dprofs = d;
        cs_fftw_plan_lock();
        db->plan_d = fftw_plan_many_dft_c2r(1, &n2, Nprof,
                                            (fftw_complex *)ws->ftprofs, NULL, k, nh + 1,
                                            d, NULL, 1, db->nbins, flags);
        if (!db->plan_d)
            db->plan_d = fftw_plan_many_dft_c2r(1, &n2, Nprof,
                                                (fftw_complex *)ws->ftprofs, NULL, k, nh + 1,
                                                d, NULL, 1, db->nbins, FFTW_ESTIMATE);
        cs_fftw_plan_unlock();
        if (!db->plan_d) return -1;
        db->psum = malloc((size_t)(db->nbins + db->wmax + 1) * sizeof(double));
        db->bcsig = malloc((size_t)(db->nbins * Nprof < CS_BOXCAR_SIGMA_SAMPLES
                                    ? db->nbins * Nprof : CS_BOXCAR_SIGMA_SAMPLES) * sizeof(double));
        db->sigcap = db->nbins * Nprof < CS_BOXCAR_SIGMA_SAMPLES
                     ? db->nbins * Nprof : CS_BOXCAR_SIGMA_SAMPLES;
    } else {
        float *d = (float *)fftwf_malloc((size_t)db->nbins * Nprof * sizeof(float));
        if (!d) return -1;
        db->dprofs = d;
        cs_fftw_plan_lock();
        db->plan_f = fftwf_plan_many_dft_c2r(1, &n2, Nprof,
                                             (fftwf_complex *)ws->ftprofs, NULL, k, nh + 1,
                                             d, NULL, 1, db->nbins, flags);
        if (!db->plan_f)
            db->plan_f = fftwf_plan_many_dft_c2r(1, &n2, Nprof,
                                                 (fftwf_complex *)ws->ftprofs, NULL, k, nh + 1,
                                                 d, NULL, 1, db->nbins, FFTW_ESTIMATE);
        cs_fftw_plan_unlock();
        if (!db->plan_f) return -1;
        db->psum = malloc((size_t)(db->nbins + db->wmax + 1) * sizeof(float));
        db->bcsig = malloc((size_t)(db->nbins * Nprof < CS_BOXCAR_SIGMA_SAMPLES
                                    ? db->nbins * Nprof : CS_BOXCAR_SIGMA_SAMPLES) * sizeof(float));
        db->sigcap = db->nbins * Nprof < CS_BOXCAR_SIGMA_SAMPLES
                     ? db->nbins * Nprof : CS_BOXCAR_SIGMA_SAMPLES;
    }
    db->shape = (double *)malloc((size_t)db->nw * sizeof(double));
    db->shapeH = -1;
    if (!db->psum || !db->bcsig || !db->shape) return -1;
    if (boxcar_batch_init(&db->batch, db->nbins, db->wmax, Nprof) != 0) return -1;
    db->src = (uint8_t *)ws->ftprofs;
    return 0;
}

static void free_decim(DecimBuf *db, int is_f64)
{
    if (!db) return;
    if (is_f64) {
        cs_fftw_plan_lock();
        if (db->plan_d) fftw_destroy_plan(db->plan_d);
        cs_fftw_plan_unlock();
        if (db->dprofs) fftw_free(db->dprofs);
    } else {
        cs_fftw_plan_lock();
        if (db->plan_f) fftwf_destroy_plan(db->plan_f);
        cs_fftw_plan_unlock();
        if (db->dprofs) fftwf_free(db->dprofs);
    }
    free(db->widths);
    free(db->shape);
    free(db->psum);
    free(db->bcsig);
    boxcar_batch_free(&db->batch);
    memset(db, 0, sizeof(*db));
}

static int workspace_init(Workspace *ws, const SearchParams *p, int Nprof,
                          unsigned flags)
{
    memset(ws, 0, sizeof(*ws));
    ws->params = (SearchParams *)p;
    ws->Nprof = Nprof;
    ws->is_f64 = (p->precision == CS_PREC_F64);
    const int nh = p->nharms;
    const int nbins = 2 * nh;
    int n2 = nbins;

    if (ws->is_f64) {
        ws->ftprofs = fftw_malloc((size_t)(nh + 1) * Nprof * sizeof(fftw_complex));
        ws->profs = fftw_malloc((size_t)nbins * Nprof * sizeof(double));
        if (!ws->ftprofs || !ws->profs) return -1;
        memset(ws->ftprofs, 0, (size_t)(nh + 1) * Nprof * sizeof(fftw_complex));
        cs_fftw_plan_lock();
        ws->base_plan_d = fftw_plan_many_dft_c2r(1, &n2, Nprof,
                                                 (fftw_complex *)ws->ftprofs, NULL, 1, nh + 1,
                                                 (double *)ws->profs, NULL, 1, nbins, flags);
        if (!ws->base_plan_d)
            ws->base_plan_d = fftw_plan_many_dft_c2r(1, &n2, Nprof,
                                                     (fftw_complex *)ws->ftprofs, NULL, 1, nh + 1,
                                                     (double *)ws->profs, NULL, 1, nbins,
                                                     FFTW_ESTIMATE);
        cs_fftw_plan_unlock();
        if (!ws->base_plan_d) return -1;
        ws->psum = malloc((size_t)(nbins + 1) * sizeof(double));
        ws->bcsig = malloc((size_t)(nbins * Nprof < CS_BOXCAR_SIGMA_SAMPLES
                                    ? nbins * Nprof : CS_BOXCAR_SIGMA_SAMPLES) * sizeof(double));
        ws->sigcap = nbins * Nprof < CS_BOXCAR_SIGMA_SAMPLES ? nbins * Nprof : CS_BOXCAR_SIGMA_SAMPLES;
    } else {
        ws->ftprofs = fftwf_malloc((size_t)(nh + 1) * Nprof * sizeof(fftwf_complex));
        ws->profs = fftwf_malloc((size_t)nbins * Nprof * sizeof(float));
        if (!ws->ftprofs || !ws->profs) return -1;
        memset(ws->ftprofs, 0, (size_t)(nh + 1) * Nprof * sizeof(fftwf_complex));
        cs_fftw_plan_lock();
        ws->base_plan_f = fftwf_plan_many_dft_c2r(1, &n2, Nprof,
                                                  (fftwf_complex *)ws->ftprofs, NULL, 1, nh + 1,
                                                  (float *)ws->profs, NULL, 1, nbins, flags);
        if (!ws->base_plan_f)
            ws->base_plan_f = fftwf_plan_many_dft_c2r(1, &n2, Nprof,
                                                      (fftwf_complex *)ws->ftprofs, NULL, 1, nh + 1,
                                                      (float *)ws->profs, NULL, 1, nbins,
                                                      FFTW_ESTIMATE);
        cs_fftw_plan_unlock();
        if (!ws->base_plan_f) return -1;
        ws->psum = malloc((size_t)(nbins + 1) * sizeof(float));
        ws->bcsig = malloc((size_t)(nbins * Nprof < CS_BOXCAR_SIGMA_SAMPLES
                                    ? nbins * Nprof : CS_BOXCAR_SIGMA_SAMPLES) * sizeof(float));
        ws->sigcap = nbins * Nprof < CS_BOXCAR_SIGMA_SAMPLES ? nbins * Nprof : CS_BOXCAR_SIGMA_SAMPLES;
    }

    ws->widths = (int *)malloc(256 * sizeof(int));
    ws->nw = ladder_boxcar_widths(nbins, 1, p, ws->widths, 256);
    ws->wmax = ws->widths[ws->nw - 1];
    /* psum was sized for the full bank; reallocate if the ladder widened it. */
    if (ws->wmax + nbins + 1 > nbins + 1) {
        if (ws->is_f64) {
            free(ws->psum);
            ws->psum = malloc((size_t)(nbins + ws->wmax + 1) * sizeof(double));
        } else {
            free(ws->psum);
            ws->psum = malloc((size_t)(nbins + ws->wmax + 1) * sizeof(float));
        }
    }
    ws->shape = (double *)malloc((size_t)ws->nw * sizeof(double));
    ws->shapeH = -1;
    if (!ws->widths || !ws->psum || !ws->bcsig || !ws->shape) return -1;
    if (boxcar_batch_init(&ws->batch, nbins, ws->wmax, Nprof) != 0) return -1;

    ws->ndecim = 0;
    { int i; for (i = 0; i < p->ndecim; i++) if (p->decimations[i] > 1) ws->ndecim++; }
    ws->decims = (DecimBuf *)calloc((size_t)(ws->ndecim ? ws->ndecim : 1), sizeof(DecimBuf));
    if (!ws->decims) return -1;
    int di = 0;
    { int i; for (i = 0; i < p->ndecim; i++) {
        int k = p->decimations[i];
        if (k <= 1) continue;
        if (build_decim(&ws->decims[di], k, ws, p, flags) != 0) return -1;
        di++;
    } }
    ws->planesize = (size_t)direct_window_size(p, Nprof);
    ws->re = (float *)malloc(ws->planesize * sizeof(float));
    ws->im = (float *)malloc(ws->planesize * sizeof(float));
    ws->filled = (uint8_t *)malloc((size_t)nh);
    if (!ws->re || !ws->im || !ws->filled) return -1;
    return 0;
}

static void workspace_free(Workspace *ws)
{
    if (!ws) return;
    int is_f64 = ws->is_f64;
    if (is_f64) {
        cs_fftw_plan_lock();
        if (ws->base_plan_d) fftw_destroy_plan(ws->base_plan_d);
        cs_fftw_plan_unlock();
        if (ws->ftprofs) fftw_free(ws->ftprofs);
        if (ws->profs) fftw_free(ws->profs);
    } else {
        cs_fftw_plan_lock();
        if (ws->base_plan_f) fftwf_destroy_plan(ws->base_plan_f);
        cs_fftw_plan_unlock();
        if (ws->ftprofs) fftwf_free(ws->ftprofs);
        if (ws->profs) fftwf_free(ws->profs);
    }
    { int i; for (i = 0; i < ws->ndecim; i++) free_decim(&ws->decims[i], is_f64); }
    free(ws->decims);
    free(ws->widths);
    free(ws->shape);
    free(ws->psum);
    free(ws->bcsig);
    boxcar_batch_free(&ws->batch);
    free(ws->re);
    free(ws->im);
    free(ws->filled);
    memset(ws, 0, sizeof(*ws));
}

Workspace *workspaces_create(const SearchParams *p, int Nprof, int nthreads,
                             unsigned plan_flags)
{
    Workspace *ws = (Workspace *)calloc((size_t)nthreads, sizeof(Workspace));
    if (!ws) return NULL;
    unsigned flags = plan_flags | FFTW_PRESERVE_INPUT;
    { int i; for (i = 0; i < nthreads; i++) {
        if (workspace_init(&ws[i], p, Nprof, flags) != 0) {
            fprintf(stderr, "Error: failed to build search workspace %d (FFTW plan failure)\n", i);
            { int j; for (j = 0; j <= i; j++) workspace_free(&ws[j]); }
            free(ws);
            return NULL;
        }
    } }
    return ws;
}

void workspaces_free(Workspace *ws, int nthreads, int ndecim)
{
    (void)ndecim;
    if (!ws) return;
    { int i; for (i = 0; i < nthreads; i++) workspace_free(&ws[i]); }
    free(ws);
}

void transform_base(Workspace *ws)
{
    if (ws->is_f64) fftw_execute(ws->base_plan_d);
    else fftwf_execute(ws->base_plan_f);
}

void transform_decim(Workspace *ws, DecimBuf *db)
{
    if (ws->is_f64) fftw_execute(db->plan_d);
    else fftwf_execute(db->plan_f);
}

/* ------------------------------------------------------------------ */
/* Metric statistics helpers                                          */
/* ------------------------------------------------------------------ */
void hist_init(MetricHistogram *h, int k, int Hk, int win, double flo, double fhi,
               double lo, double hi, int nb)
{
    memset(h, 0, sizeof(*h));
    h->k = k; h->Hk = Hk; h->nbins = 2 * Hk; h->win = win;
    h->flo = flo; h->fhi = fhi; h->lo = lo; h->hi = hi;
    h->invw = (double)nb / (hi - lo);
    h->nb = nb;
    h->counts = (long long *)calloc((size_t)nb, sizeof(long long));
    h->vmin = INFINITY;
    h->vmax = -INFINITY;
}
void hist_free(MetricHistogram *h)
{
    if (!h) return;
    free(h->counts);
    memset(h, 0, sizeof(*h));
}
void hist_push(MetricHistogram *h, double x)
{
    h->total++;
    h->sum += x;
    h->sumsq += x * x;
    if (x < h->vmin) h->vmin = x;
    if (x > h->vmax) h->vmax = x;
    if (x < h->lo) h->under++;
    else if (x >= h->hi) h->over++;
    else {
        int idx = (int)floor((x - h->lo) * h->invw);
        if (idx < 0) idx = 0;
        if (idx >= h->nb) idx = h->nb - 1;
        h->counts[idx]++;
    }
}
void hist_merge(MetricHistogram *a, const MetricHistogram *b)
{
    { int i; for (i = 0; i < a->nb; i++) a->counts[i] += b->counts[i]; }
    a->under += b->under; a->over += b->over; a->total += b->total;
    a->sum += b->sum; a->sumsq += b->sumsq;
    if (b->vmin < a->vmin) a->vmin = b->vmin;
    if (b->vmax > a->vmax) a->vmax = b->vmax;
}
double hist_quantile(const MetricHistogram *h, double q)
{
    if (h->total == 0) return NAN;
    double target = q * (double)h->total;
    long long cum = h->under;
    if ((double)cum >= target) return h->lo;
    double w = 1.0 / h->invw;
    { int i; for (i = 0; i < h->nb; i++) {
        long long c = h->counts[i];
        if ((double)(cum + c) >= target) {
            double edge = h->lo + (double)i * w;
            return c == 0 ? edge : edge + (target - (double)cum) / (double)c * w;
        }
        cum += c;
    } }
    return h->hi;
}
void log_edges(double a, double b, int nwin, double *edges)
{
    if (a < DBL_MIN) a = DBL_MIN;
    if (b < a * (1.0 + 1e-9)) b = a * (1.0 + 1e-9);
    double la = log10(a), lb = log10(b);
    { int i; for (i = 0; i <= nwin; i++)
        edges[i] = pow(10.0, la + (lb - la) * (double)i / (double)nwin); }
}
int window_index(const double *edges, int nwin, double f)
{
    int lo = 0, hi = nwin + 1; /* number of edges <= f in [0,nwin+1) */
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (edges[mid] <= f) lo = mid + 1;
        else hi = mid;
    }
    int idx = lo; /* 1-based count */
    if (idx < 1) idx = 1;
    if (idx > nwin) idx = nwin;
    return idx;
}

void metricstats_init(MetricStats *ms, double lo, double hi, int nb, int nwin)
{
    memset(ms, 0, sizeof(*ms));
    ms->hist_lo = lo; ms->hist_hi = hi; ms->hist_nb = nb;
    ms->nwin = nwin < 1 ? 1 : nwin;
}
void metricstats_free(MetricStats *ms)
{
    { size_t i; for (i = 0; i < ms->nblocks; i++) (void)i; }
    free(ms->blocks);
    { size_t i; for (i = 0; i < ms->nhists; i++) hist_free(&ms->hists[i]); }
    free(ms->hists);
    { size_t i; for (i = 0; i < ms->nwhists; i++) hist_free(&ms->whists[i]); }
    free(ms->whists);
    memset(ms, 0, sizeof(*ms));
}
void metricstats_append_block(MetricStats *ms, BlockMetricStats s)
{
    if (ms->nblocks == ms->capblocks) {
        size_t nc = ms->capblocks ? ms->capblocks * 2 : 256;
        ms->blocks = (BlockMetricStats *)realloc(ms->blocks, nc * sizeof(BlockMetricStats));
        ms->capblocks = nc;
    }
    ms->blocks[ms->nblocks++] = s;
}
void metricstats_append_hist(MetricStats *ms, MetricHistogram h)
{
    if (ms->nhists == ms->caphists) {
        size_t nc = ms->caphists ? ms->caphists * 2 : 16;
        ms->hists = (MetricHistogram *)realloc(ms->hists, nc * sizeof(MetricHistogram));
        ms->caphists = nc;
    }
    ms->hists[ms->nhists++] = h;
}
void metricstats_append_whist(MetricStats *ms, MetricHistogram h)
{
    if (ms->nwhists == ms->capwhists) {
        size_t nc = ms->capwhists ? ms->capwhists * 2 : 64;
        ms->whists = (MetricHistogram *)realloc(ms->whists, nc * sizeof(MetricHistogram));
        ms->capwhists = nc;
    }
    ms->whists[ms->nwhists++] = h;
}

static BlockMetricStats block_stats(int block, int k, int Hk, int nbins,
                                    double ngoodbins, double flo, double fhi,
                                    double *v, long long n)
{
    BlockMetricStats s;
    memset(&s, 0, sizeof(s));
    s.block = block; s.k = k; s.Hk = Hk; s.nbins = nbins;
    s.ngoodbins = ngoodbins; s.flo = flo; s.fhi = fhi; s.n = n;
    if (n == 0) {
        s.min = s.median = s.mean = s.std = s.max = NAN;
        return s;
    }
    double vmin = v[0], vmax = v[0], sum = 0.0;
    { long long i; for (i = 0; i < n; i++) {
        double x = v[i];
        if (x < vmin) vmin = x;
        if (x > vmax) vmax = x;
        sum += x;
    } }
    double mean = sum / (double)n;
    double ss = 0.0;
    { long long i; for (i = 0; i < n; i++) {
        double d = v[i] - mean;
        ss += d * d;
    } }
    double std = n > 1 ? sqrt(ss / (double)(n - 1)) : 0.0;
    double med = median_f64(v, (int)n);
    s.min = vmin; s.median = med; s.mean = mean; s.std = std; s.max = vmax;
    return s;
}

/* ------------------------------------------------------------------ */
/* Per-thread results                                                 */
/* ------------------------------------------------------------------ */
typedef struct {
    CandidateList out;
    BlockMetricStats *stats;
    size_t nstats, capstats;
    MetricHistogram *hists; /* nstatks * nwin */
    int nstatks;
    int nwin;
    double t_interp, t_brfft, t_metric, t_decim_brfft, t_decim_metric;
    double t_busy;
    long long nchunks;
} ThreadResult;

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void stats_push(ThreadResult *tr, BlockMetricStats s)
{
    if (tr->nstats == tr->capstats) {
        size_t nc = tr->capstats ? tr->capstats * 2 : 256;
        tr->stats = (BlockMetricStats *)realloc(tr->stats, nc * sizeof(BlockMetricStats));
        tr->capstats = nc;
    }
    tr->stats[tr->nstats++] = s;
}

static void tr_free(ThreadResult *tr)
{
    candlist_free(&tr->out);
    free(tr->stats);
    if (tr->hists) {
        { int i; for (i = 0; i < tr->nstatks * tr->nwin; i++) hist_free(&tr->hists[i]); }
        free(tr->hists);
    }
    memset(tr, 0, sizeof(*tr));
}

/* ------------------------------------------------------------------ */
/* Decimation pass                                                    */
/* ------------------------------------------------------------------ */
static void decim_pass(CandidateList *out, Workspace *ws, DecimBuf *db,
                       const FFTFile *ft, const SearchParams *p,
                       double rstart, double lodr, int n,
                       double threshold, long long block, int collect,
                       ThreadResult *tr,
                       MetricHistogram *hist, const MetricNorm *norm,
                       double exactcut)
{
    const int k = db->k, Hk = db->Hk, nbins = db->nbins;
    double _t0 = now_sec();
    transform_decim(ws, db);
    tr->t_decim_brfft += now_sec() - _t0;

    double rmean = rstart + (double)(n - 1) * lodr / 2.0;
    double nyq = (double)ft->N / 2.0;
    int nvalid = 0;
    while (nvalid < n && (double)k * (rstart + (double)nvalid * lodr) < nyq) nvalid++;

    double sig_d;
    if (p->sigma == CS_SIGMA_ANALYTIC)
        sig_d = analytic_sigma(ws->filled, k, Hk);
    else {
        if (ws->is_f64)
            sig_d = block_sigma_f64((double *)db->dprofs, nbins, nvalid,
                                    (double *)db->bcsig, db->sigcap);
        else
            sig_d = block_sigma_f32((float *)db->dprofs, nbins, nvalid,
                                    (float *)db->bcsig, db->sigcap);
    }

    double *mbuf = NULL;
    long long nmb = 0;
    if (collect) {
        mbuf = (double *)malloc((size_t)(nvalid ? nvalid : 1) * sizeof(double));
    }

    double _t1 = now_sec();
    if (ws->is_f64) {
        double sig = sig_d;
        double invsigma = sig > 0.0 ? 1.0 / sig : 0.0;
        refresh_shape(db->shape, &db->shapeH, nbins, db->widths, db->nw,
                      ws->filled, k, Hk);
        boxcar_metrics_f64(&db->batch, (double *)db->dprofs, nvalid,
                           (double *)db->psum, db->widths, db->nw, nbins,
                           invsigma, db->shape, exactcut);
    } else {
        float sig = (float)sig_d;
        float invsigma = sig > 0.0f ? 1.0f / sig : 0.0f;
        refresh_shape(db->shape, &db->shapeH, nbins, db->widths, db->nw,
                      ws->filled, k, Hk);
        boxcar_metrics_f32(&db->batch, (float *)db->dprofs, nvalid,
                           (float *)db->psum, db->widths, db->nw, nbins,
                           invsigma, db->shape, exactcut);
    }
    tr->t_decim_metric += now_sec() - _t1;

    { int j; for (j = 0; j < n; j++) {
        double r_dec = (double)k * (rstart + (double)j * lodr);
        if (!(r_dec < nyq)) continue;
        double mval = db->batch.mvals[j];
        if (collect) {
            mbuf[nmb++] = mval;
            if (hist) hist_push(hist, mval);
        }
        double fdec = r_dec / ft->T;
        double score = (norm == NULL) ? mval : metric_normalize(norm, k, fdec, mval);
        if (score > threshold) {
            Candidate c;
            c.freq = fdec; c.metric = score; c.r = r_dec; c.nharm = Hk;
            c.ducy = NAN;
            candlist_push(out, c);
        }
    } }

    if (collect && nmb > 0) {
        double flo = (double)k * rstart / ft->T;
        double fhi = (double)k * (rstart + (double)(nmb - 1) * lodr) / ft->T;
        double ngood = (double)ft->N / 2.0 / (k * rmean);
        if (ngood > (double)Hk) ngood = (double)Hk;
        BlockMetricStats s = block_stats((int)block, k, Hk, nbins, ngood,
                                         flo, fhi, mbuf, nmb);
        stats_push(tr, s);
    }
    free(mbuf);
}

/* ------------------------------------------------------------------ */
/* Normalisation model                                                */
/* ------------------------------------------------------------------ */
static void loc_scale(const MetricHistogram *h, double *loc, double *scale)
{
    *loc = hist_quantile(h, 0.5);
    double hi = hist_quantile(h, 0.8413);
    *scale = hi - *loc;
    if (!(*scale > 0.0)) *scale = NAN;
}

MetricNorm *build_metricnorm(const MetricStats *ms)
{
    MetricNorm *nm = (MetricNorm *)calloc(1, sizeof(MetricNorm));
    if (!nm) return NULL;
    nm->ks = (NormK *)calloc(ms->nhists ? ms->nhists : 1, sizeof(NormK));
    if (!nm->ks) { free(nm); return NULL; }
    { size_t hi; for (hi = 0; hi < ms->nhists; hi++) {
        const MetricHistogram *g = &ms->hists[hi];
        int k = g->k;
        double gloc, gscale;
        loc_scale(g, &gloc, &gscale);
        if (isnan(gscale)) gscale = 1.0;
        int nwin = 0;
        { size_t wi; for (wi = 0; wi < ms->nwhists; wi++) if (ms->whists[wi].k == k) nwin++; }
        if (nwin == 0) continue;
        NormK *nk = &nm->ks[nm->n];
        nk->k = k; nk->nwin = nwin;
        nk->edges = (double *)malloc((size_t)(nwin + 1) * sizeof(double));
        nk->loc = (double *)malloc((size_t)nwin * sizeof(double));
        nk->scale = (double *)malloc((size_t)nwin * sizeof(double));
        int wi = 0;
        double last_fhi = 0.0;
        { size_t w; for (w = 0; w < ms->nwhists; w++) {
            if (ms->whists[w].k != k) continue;
            const MetricHistogram *h = &ms->whists[w];
            nk->edges[wi] = h->flo;
            last_fhi = h->fhi;
            double wloc, wscale;
            loc_scale(h, &wloc, &wscale);
            if (h->total < CS_MIN_WIN_TRIALS || isnan(wscale)) {
                nk->loc[wi] = gloc; nk->scale[wi] = gscale;
            } else {
                nk->loc[wi] = wloc; nk->scale[wi] = wscale;
            }
            wi++;
        } }
        nk->edges[nwin] = last_fhi;
        nm->n++;
    } }
    return nm;
}

void metricnorm_free(MetricNorm *n)
{
    if (!n) return;
    { int i; for (i = 0; i < n->n; i++) {
        free(n->ks[i].edges);
        free(n->ks[i].loc);
        free(n->ks[i].scale);
    } }
    free(n->ks);
    free(n);
}

double metric_normalize(const MetricNorm *n, int k, double f, double M)
{
    { int i; for (i = 0; i < n->n; i++) {
        if (n->ks[i].k != k) continue;
        int w = window_index(n->ks[i].edges, n->ks[i].nwin, f);
        return (M - n->ks[i].loc[w - 1]) / n->ks[i].scale[w - 1];
    } }
    return M;
}

/* ------------------------------------------------------------------ */
/* Threaded chunk region                                              */
/* ------------------------------------------------------------------ */
static void render_progress(int mode, long long done, long long total)
{
    if (mode == 0) return;
    double frac = total == 0 ? 1.0 : (double)done / (double)total;
    int pct = (int)llround(100.0 * frac);
    if (mode == 2) {
        const int width = 40;
        int filled = (int)llround(width * frac);
        if (filled < 0) filled = 0;
        if (filled > width) filled = width;
        fprintf(stderr, "\r  Searching [");
        { int i; for (i = 0; i < filled; i++) fputc('#', stderr); }
        { int i; for (i = filled; i < width; i++) fputc(' ', stderr); }
        fprintf(stderr, "] %3d%%  (%lld/%lld chunks)", pct, done, total);
    } else {
        fprintf(stderr, "\r  Searching: %3d%%  (%lld/%lld chunks)", pct, done, total);
    }
    fflush(stderr);
}

/* Collects one chunk's above-threshold candidates and (optionally) statistics. */
static void process_chunk(Workspace *ws, ThreadResult *tr, const DirectPlanSet *dplans,
                          const FFTFile *ft, const SearchParams *p,
                          double r_lo, double lodr, long long total,
                          long long c, long long nchunks, const long long *cstarts,
                          int Nprof, int collect, MetricStats *metricstats,
                          const int *statks, int nstatks, int nwin,
                          const double **wedges, double threshold,
                          const MetricNorm *norm, double exactcut,
                          int use_analytic)
{
    const int nh = p->nharms;
    const int nbins = 2 * nh;
    long long i0 = cstarts[c];
    int n = (int)((c + 1 < nchunks ? cstarts[c + 1] : total) - i0);
    double rstart = r_lo + (double)i0 * lodr;

    double _t0 = now_sec();
    fill_chunk_profiles(ws, dplans, ft, p, rstart, lodr, n, i0);
    tr->t_interp += now_sec() - _t0;
    _t0 = now_sec();
    transform_base(ws);
    tr->t_brfft += now_sec() - _t0;
    double rmean = rstart + (double)(n - 1) * lodr / 2.0;

    _t0 = now_sec();
    if (ws->is_f64) {
        double sig = use_analytic ? analytic_sigma(ws->filled, 1, nh)
                                    : block_sigma_f64((double *)ws->profs, nbins, n,
                                                      (double *)ws->bcsig, ws->sigcap);
        double invsigma = sig > 0.0 ? 1.0 / sig : 0.0;
        refresh_shape(ws->shape, &ws->shapeH, nbins, ws->widths, ws->nw,
                      ws->filled, 1, nh);
        boxcar_metrics_f64(&ws->batch, (double *)ws->profs, n, (double *)ws->psum,
                           ws->widths, ws->nw, nbins, invsigma, ws->shape, exactcut);
    } else {
        float sig = use_analytic ? (float)analytic_sigma(ws->filled, 1, nh)
                                   : block_sigma_f32((float *)ws->profs, nbins, n,
                                                     (float *)ws->bcsig, ws->sigcap);
        float invsigma = sig > 0.0f ? 1.0f / sig : 0.0f;
        refresh_shape(ws->shape, &ws->shapeH, nbins, ws->widths, ws->nw,
                      ws->filled, 1, nh);
        boxcar_metrics_f32(&ws->batch, (float *)ws->profs, n, (float *)ws->psum,
                           ws->widths, ws->nw, nbins, invsigma, ws->shape, exactcut);
    }
    tr->t_metric += now_sec() - _t0;

    MetricHistogram *basehist = NULL;
    if (collect) {
        int w = window_index(wedges[0], nwin, rmean / ft->T);
        basehist = &tr->hists[w - 1];
    }
    { int j; for (j = 0; j < n; j++) {
        double metric = ws->batch.mvals[j];
        if (collect) hist_push(basehist, metric);
        double rf = rstart + (double)j * lodr;
        double score = (norm == NULL) ? metric : metric_normalize(norm, 1, rf / ft->T, metric);
        if (score > threshold) {
            Candidate cd;
            cd.freq = rf / ft->T; cd.metric = score; cd.r = rf; cd.nharm = nh;
            cd.ducy = NAN;
            candlist_push(&tr->out, cd);
        }
    } }
    if (collect) {
        double flo = rstart / ft->T;
        double fhi = (rstart + (double)(n - 1) * lodr) / ft->T;
        double ngood = (double)ft->N / 2.0 / rmean;
        if (ngood > (double)nh) ngood = (double)nh;
        double *mbuf = (double *)malloc((size_t)n * sizeof(double));
        { int j; for (j = 0; j < n; j++) mbuf[j] = ws->batch.mvals[j]; }
        BlockMetricStats s = block_stats((int)c + 1, 1, nh, nbins, ngood, flo, fhi,
                                         mbuf, n);
        free(mbuf);
        stats_push(tr, s);
    }

    { int di; for (di = 0; di < ws->ndecim; di++) {
        DecimBuf *db = &ws->decims[di];
        MetricHistogram *whist = NULL;
        if (collect) {
            int si = 0;
            while (si < nstatks - 1 && statks[si] < db->k) si++;
            int w = window_index(wedges[si], nwin, (double)db->k * rmean / ft->T);
            whist = &tr->hists[si * nwin + (w - 1)];
        }
        decim_pass(&tr->out, ws, db, ft, p, rstart, lodr, n, threshold,
                   c + 1, collect, tr, whist, norm, exactcut);
    } }
}

static CandidateList search_region(const FFTFile *ft, const SearchParams *p,
                                   Workspace *workspaces, int nbins,
                                   double r_lo, double r_hi, double lodr,
                                   long long total, int Nprof, long long nchunks,
                                   int nt, const long long *cstarts,
                                   double threshold, const MetricNorm *norm,
                                   MetricStats *metricstats, int progress_mode,
                                   const DirectPlanSet *dplans)
{
    CandidateList result; candlist_init(&result);
    int collect = metricstats != NULL;
    int analytic_sigma = (p->sigma == CS_SIGMA_ANALYTIC);
    double exactcut = (!collect && norm == NULL) ? threshold - p->boxcar_gatemargin
                                                 : -INFINITY;

    int statks[64]; int nstatks = 0;
    double *wedges[64];
    int nwin = collect ? metricstats->nwin : 0;
    if (collect) {
        statks[nstatks++] = 1;
        { int i; for (i = 0; i < workspaces[0].ndecim; i++) {
            int k = workspaces[0].decims[i].k;
            int dup = 0;
            { int j; for (j = 0; j < nstatks; j++) if (statks[j] == k) dup = 1; }
            if (!dup) statks[nstatks++] = k;
        } }
        qsort(statks, (size_t)nstatks, sizeof(int), int_cmp_local);
        { int i; for (i = 0; i < nstatks; i++) {
            wedges[i] = (double *)malloc((size_t)(nwin + 1) * sizeof(double));
            log_edges((double)statks[i] * r_lo / ft->T,
                      (double)statks[i] * r_hi / ft->T, nwin, wedges[i]);
        } }
    }

    ThreadResult *tr = (ThreadResult *)calloc((size_t)nt, sizeof(ThreadResult));
    { int t; for (t = 0; t < nt; t++) {
        candlist_init(&tr[t].out);
        tr[t].nstatks = nstatks;
        tr[t].nwin = nwin;
        if (collect) {
            tr[t].hists = (MetricHistogram *)calloc((size_t)nstatks * nwin,
                                                    sizeof(MetricHistogram));
            { int si; for (si = 0; si < nstatks; si++) {
                int k = statks[si];
                int Hk = p->nharms / k;
                { int w; for (w = 0; w < nwin; w++) {
                    hist_init(&tr[t].hists[si * nwin + w], k, Hk, w + 1,
                              wedges[si][w], wedges[si][w + 1],
                              metricstats->hist_lo, metricstats->hist_hi,
                              metricstats->hist_nb);
                } }
            } }
        }
    } }

    long long done = 0;
    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        ThreadResult *mtr = &tr[tid];
        { long long c; for (c = tid; c < nchunks; c += nt) {
            double _tb = now_sec();
            process_chunk(&workspaces[tid], mtr, dplans, ft, p, r_lo, lodr,
                          total, c, nchunks, cstarts, Nprof, collect,
                          metricstats, statks, nstatks, nwin,
                          (const double **)wedges, threshold, norm, exactcut,
                          analytic_sigma);
            mtr->t_busy += now_sec() - _tb;
            mtr->nchunks++;
            long long d = __atomic_add_fetch(&done, 1, __ATOMIC_RELAXED);
            if (tid == 0) render_progress(progress_mode, d, nchunks);
        } }
    }
    if (progress_mode != 0) {
        render_progress(progress_mode, nchunks, nchunks);
        fprintf(stderr, "\n");
    }

    { int t; for (t = 0; t < nt; t++) candlist_append(&result, &tr[t].out); }

    if (getenv("CS_PHASES")) {
        double ti = 0, tb = 0, tm = 0, td = 0, tdm = 0;
        { int t; for (t = 0; t < nt; t++) {
            ti += tr[t].t_interp; tb += tr[t].t_brfft; tm += tr[t].t_metric;
            td += tr[t].t_decim_brfft; tdm += tr[t].t_decim_metric;
        } }
        double gt = 0, gs = 0;
        if (nt == 1) boxcar_gate_times(&gt, &gs);
        fprintf(stderr, "PHASES interp=%.3f brfft=%.3f metric=%.3f decim-brfft=%.3f decim-metric=%.3f total=%.3f\n",
                ti, tb, tm, td, tdm, ti + tb + tm + td + tdm);
        if (nt == 1) fprintf(stderr, "GATE transpose=%.3f scan=%.3f\n", gt, gs);
        double bmin = 1e30, bmax = 0, bsum = 0;
        { int t; for (t = 0; t < nt; t++) {
            if (tr[t].t_busy < bmin) bmin = tr[t].t_busy;
            if (tr[t].t_busy > bmax) bmax = tr[t].t_busy;
            bsum += tr[t].t_busy;
        } }
        fprintf(stderr, "THREADS n=%d busy min=%.3f max=%.3f avg=%.3f sum=%.3f\n",
                nt, bmin, bmax, bsum / nt, bsum);
    }

    if (collect) {
        { int t; for (t = 0; t < nt; t++) {
            { size_t i; for (i = 0; i < tr[t].nstats; i++)
                metricstats_append_block(metricstats, tr[t].stats[i]); }
        } }
        { int si; for (si = 0; si < nstatks; si++) {
            MetricHistogram *kwins = (MetricHistogram *)calloc((size_t)nwin,
                                                               sizeof(MetricHistogram));
            { int w; for (w = 0; w < nwin; w++) {
                kwins[w] = tr[0].hists[si * nwin + w];
                tr[0].hists[si * nwin + w].counts = NULL;
                { int t; for (t = 1; t < nt; t++)
                    hist_merge(&kwins[w], &tr[t].hists[si * nwin + w]); }
            } }
            { int w; for (w = 0; w < nwin; w++)
                metricstats_append_whist(metricstats, kwins[w]); }
            /* band-wide merge */
            MetricHistogram g;
            hist_init(&g, statks[si], p->nharms / statks[si], 0,
                      wedges[si][0], wedges[si][nwin],
                      metricstats->hist_lo, metricstats->hist_hi,
                      metricstats->hist_nb);
            { int w; for (w = 0; w < nwin; w++) hist_merge(&g, &kwins[w]); }
            metricstats_append_hist(metricstats, g);
            free(kwins);
        } }
        /* sort blocks by (block,k) */
        qsort(metricstats->blocks, metricstats->nblocks,
              sizeof(BlockMetricStats), block_cmp);
    }

    { int t; for (t = 0; t < nt; t++) tr_free(&tr[t]); }
    free(tr);
    { int i; for (i = 0; i < nstatks; i++) free(wedges[i]); }
    return result;
}

/* ------------------------------------------------------------------ */
/* Analytic-sigma sanity check                                        */
/* ------------------------------------------------------------------ */
static int sigma_sanity_check(const FFTFile *ft, const SearchParams *p,
                              Workspace *ws, const DirectPlanSet *dplans,
                              double r_lo, double lodr, long long total, int Nprof)
{
    long long nchunks = (total + Nprof - 1) / Nprof;
    int nsample = (int)(CS_SIGMA_CHECK_CHUNKS < nchunks ? CS_SIGMA_CHECK_CHUNKS : nchunks);
    double worst = 0.0, meas_w = 0.0, ana_w = 0.0, worst_f = 0.0;
    int worst_k = 1;
    const int nbins = 2 * p->nharms;
    { int i; for (i = 0; i < nsample; i++) {
        long long c = (nsample == 1) ? 1 : 1 + ((long long)i * (nchunks - 1)) / (nsample - 1);
        long long i0 = (c - 1) * Nprof;
        int n = (int)((total - i0) < Nprof ? (total - i0) : Nprof);
        if (n < Nprof && total >= Nprof) continue;
        if (n < 8) continue;
        double rstart = r_lo + (double)i0 * lodr;
        fill_chunk_profiles(ws, dplans, ft, p, rstart, lodr, n, i0);
        transform_base(ws);
        int nk = 1 + ws->ndecim;
        { int ik; for (ik = 0; ik < nk; ik++) {
            const void *prof;
            int k, Hk, nb;
            if (ik == 0) { prof = ws->profs; k = 1; Hk = p->nharms; nb = nbins; }
            else {
                DecimBuf *db = &ws->decims[ik - 1];
                transform_decim(ws, db);
                prof = db->dprofs; k = db->k; Hk = db->Hk; nb = db->nbins;
            }
            double meas, ana;
            if (ws->is_f64)
                meas = block_sigma_f64((const double *)prof, nb, n,
                                       (double *)ws->bcsig, ws->sigcap);
            else
                meas = block_sigma_f32((const float *)prof, nb, n,
                                       (float *)ws->bcsig, ws->sigcap);
            ana = analytic_sigma(ws->filled, k, Hk);
            if (!(meas > 0.0 && ana > 0.0)) continue;
            double dev = fabs(ana / meas - 1.0);
            if (dev > worst) {
                worst = dev; worst_k = k; worst_f = rstart / ft->T;
                meas_w = meas; ana_w = ana;
            }
        } }
    } }
    if (worst > CS_SIGMA_CHECK_TOL) {
        double ratio = ana_w / meas_w;
        fprintf(stderr,
                "Warning: The analytic noise scale disagrees with the measured one by ");
        if (ratio > 2.0 || ratio < 0.5)
            fprintf(stderr, "a factor of %.4g", ratio > 1.0 ? ratio : 1.0 / ratio);
        else
            fprintf(stderr, "%.1f%%", 100.0 * worst);
        fprintf(stderr,
                " -- every reported S/N is scaled by that.  This normally means the "
                "input FFT is not normalised (Fourier powers with mean 1); run "
                "PRESTO's `rednoise` on it, or pass `--sigma measured` to estimate "
                "the scale from the data instead.\n"
                "         worst at k=%d, %.4f Hz: measured %.5g vs analytic %.5g "
                "(ratio %.4g)\n",
                worst_k, worst_f, meas_w, ana_w, ratio);
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* Wisdom                                                             */
/* ------------------------------------------------------------------ */
static char *default_wisdom_path(void)
{
    const char *env = getenv("COHERENT_WISDOM");
    if (env && *env) return strdup(env);
    const char *home = getenv("HOME");
    char *dir = NULL;
    if (home && *home) {
        size_t n = strlen(home) + 32;
        dir = (char *)malloc(n);
        snprintf(dir, n, "%s/.cache/coherent_search", home);
    } else {
        dir = strdup("/tmp/coherent_search");
    }
    if (!dir) return NULL;
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", dir);
    if (system(cmd) != 0) { /* best effort */ }
    char host[256];
    if (gethostname(host, sizeof(host)) != 0) strcpy(host, "localhost");
    size_t n = strlen(dir) + strlen(host) + 32;
    char *path = (char *)malloc(n);
    snprintf(path, n, "%s/fftw_wisdom_%s.dat", dir, host);
    free(dir);
    return path;
}

/* ------------------------------------------------------------------ */
/* Search entry point                                                 */
/* ------------------------------------------------------------------ */
static char *wisdom_path_for(int precision, const char *wisdom_file)
{
    char *base = (wisdom_file && *wisdom_file) ? strdup(wisdom_file)
                                               : default_wisdom_path();
    if (!base) return NULL;
    size_t n = strlen(base) + 8;
    char *path = (char *)malloc(n);
    snprintf(path, n, "%s.%s", base,
             precision == CS_PREC_F64 ? "f64" : "f32");
    free(base);
    return path;
}

int cs_prime_wisdom(const SearchParams *p, int Nprof, int wisdom,
                    const char *wisdom_file)
{
    if (!wisdom) {
        fprintf(stderr, "Error: --prime-wisdom requires wisdom (drop --nowisdom)\n");
        return 1;
    }
    if (Nprof < 1) Nprof = 1;
    { int ip; for (ip = 0; ip < 2; ip++) {
        int prec = (ip == 0) ? CS_PREC_F32 : CS_PREC_F64;
        char *wpath = wisdom_path_for(prec, wisdom_file);
        if (!wpath) return 1;
        cs_fftw_plan_lock();
        if (prec == CS_PREC_F64) fftw_import_wisdom_from_filename(wpath);
        else fftwf_import_wisdom_from_filename(wpath);
        cs_fftw_plan_unlock();
        SearchParams q = *p;
        q.precision = (cs_precision)prec;
        Workspace *ws = workspaces_create(&q, Nprof, 1,
                                          FFTW_PATIENT | FFTW_PRESERVE_INPUT);
        if (!ws) {
            fprintf(stderr, "Error: failed to build FFTW plans for %s\n",
                    prec == CS_PREC_F64 ? "f64" : "f32");
            free(wpath);
            return 1;
        }
        workspaces_free(ws, 1, q.nharms);
        cs_fftw_plan_lock();
        if (prec == CS_PREC_F64) fftw_export_wisdom_to_filename(wpath);
        else fftwf_export_wisdom_to_filename(wpath);
        cs_fftw_plan_unlock();
        fprintf(stderr, "Info: primed %s FFTW wisdom (nharms=%d, Nprof=%d) -> %s\n",
                prec == CS_PREC_F64 ? "f64" : "f32", q.nharms, Nprof, wpath);
        free(wpath);
    } }
    return 0;
}

CandidateList search_file(const FFTFile *ft, const SearchParams *p,
                          double lofreq, double hifreq, int lobin, int blocksize,
                          double threshold, int remove, double dr_tol,
                          int harm_remove, int numharm, int progress_mode,
                          MetricStats *metricstats, int normalize, int verbose,
                          int wisdom, const char *wisdom_file,
                          SearchCache *cache, double harm_tol,
                          int gpu_device)
{
    CandidateList empty; candlist_init(&empty);
    double lodr = p->hidr / (double)p->nharms;
    int nbins = 2 * p->nharms;
    double r_lo = lofreq * ft->T;
    if (lobin != 100) r_lo = (double)lobin;
    double r_hi = hifreq * ft->T;
    long long total = (long long)floor((r_hi - r_lo) / lodr) + 1;
    if (total < 0) total = 0;
    if (total == 0) return empty;
    int Nprof = blocksize == 0 ? CS_CPU_DEFAULT_BLOCKSIZE : blocksize;
    if (Nprof < 1) Nprof = 1;

    /* FFTW wisdom is process-global: import once per SearchCache, and skip it
     * entirely on the GPU path (which uses cuFFT). */
    int use_wisdom = wisdom && gpu_device < 0;
    char *wpath = NULL;
    if (use_wisdom && !(cache && cache->wisdom_done)) {
        wpath = wisdom_path_for(p->precision, wisdom_file);
        if (wpath) {
            cs_fftw_plan_lock();
            if (p->precision == CS_PREC_F64) fftw_import_wisdom_from_filename(wpath);
            else fftwf_import_wisdom_from_filename(wpath);
            cs_fftw_plan_unlock();
        }
    }

    /* Reuse the file-independent plans across a DM sweep.  They depend on the
     * observation length (through r_lo), the trial grid, Nprof and the searched
     * band; for a list of .dat files only the data (and DM) changes. */
    DirectPlanSet local_plans;
    memset(&local_plans, 0, sizeof(local_plans));
    DirectPlanSet *pdplans = NULL;
    long long *cstarts = NULL;
    long long nchunks = 0;
    int own_plans = 0;

    int plan_hit = cache && cache->plan_valid && cache->params == p &&
                   cache->plan_T == ft->T &&
                   cache->plan_lofreq == lofreq && cache->plan_hifreq == hifreq &&
                   cache->plan_lobin == lobin && cache->plan_Nprof == Nprof &&
                   cache->plan_total == total;
    if (plan_hit) {
        pdplans = &cache->dplans;
        cstarts = cache->cstarts;
        nchunks = cache->nchunks;
    } else {
        DirectPlanSet *target = &local_plans;
        if (cache) {
            if (cache->plan_valid) {
                direct_plans_free(&cache->dplans);
                free(cache->cstarts);
                cache->cstarts = NULL;
                cache->plan_valid = 0;
            }
            target = &cache->dplans;
        } else {
            own_plans = 1;
        }
        if (direct_plans_build(target, p, r_lo) != 0) { free(wpath); return empty; }
        long long *cs = NULL;
        long long nc = 0;
        if (chunk_starts(target, ft, p, total, Nprof, &cs, &nc) != 0) {
            if (cache) direct_plans_free(&cache->dplans);
            else direct_plans_free(target);
            free(wpath);
            return empty;
        }
        pdplans = target;
        cstarts = cs;
        nchunks = nc;
        if (cache) {
            cache->cstarts = cs;
            cache->nchunks = nc;
            cache->params = (SearchParams *)p;
            cache->plan_T = ft->T;
            cache->plan_lofreq = lofreq;
            cache->plan_hifreq = hifreq;
            cache->plan_lobin = lobin;
            cache->plan_Nprof = Nprof;
            cache->plan_total = total;
            cache->plan_valid = 1;
        }
    }

    int nt = omp_get_max_threads();
    if (nt > nchunks) nt = (int)nchunks;
    if (nt < 1) nt = 1;

    if (verbose) {
        fprintf(stderr,
                "Search: %lld trials in %lld chunks of %d over r = %.1f ... %.1f bins "
                "(%.4f ... %.4f Hz), %s\n",
                total, nchunks, Nprof, r_lo, r_hi, r_lo / ft->T, r_hi / ft->T,
                gpu_device >= 0 ? "GPU" : "CPU");
        int nofb = 0, pmin = 0, pmax = 0;
        { int i; for (i = 0; i < pdplans->n; i++) {
            if (pdplans->plans[i].P == 0) nofb++;
            else {
                if (pmin == 0 || pdplans->plans[i].P < pmin) pmin = pdplans->plans[i].P;
                if (pdplans->plans[i].P > pmax) pmax = pdplans->plans[i].P;
            }
        } }
        if (nofb > 0)
            fprintf(stderr, "  NOTE: %d harmonic(s) fell back to per-trial coefficients "
                            "(hidr=%g is not a simple rational)\n", nofb, p->hidr);
        else
            fprintf(stderr, "  phase-cycle lengths P: min %d, max %d (of q = %d)\n",
                    pmin, pmax, pdplans->plans[0].q);
        fprintf(stderr, "  interpolation: m=%d, Float32 weights (~2e-7 relative); "
                        "truncation loss ~%.2f%%, hidr grid ~6.5%%\n",
                p->m, 100.0 * 0.203 / (double)p->m);
    }

    /* The GPU interpolation kernel needs the tabulated phase/group tables; a
     * non-rational trial step falls back to the per-trial CPU kernel. */
    if (gpu_device >= 0) {
        { int i; for (i = 0; i < pdplans->n; i++) {
            if (pdplans->plans[i].P == 0) {
                fprintf(stderr, "Warning: --hidr is not a simple rational; the GPU "
                                "path needs the tabulated kernel.  Falling back to CPU.\n");
                gpu_device = -1;
                break;
            }
        } }
    }

    CandidateList cands;
    candlist_init(&cands);

    if (gpu_device >= 0) {
        GpuSession *gs = cache ? (GpuSession *)cache->gpu_session : NULL;
        int own_session = 0;
        if (!gs || !cs_gpu_session_matches(gs, ft, p, Nprof, total, r_lo,
                                           lodr, gpu_device)) {
            if (gs) cs_gpu_session_free(gs);
            gs = cs_gpu_session_create(ft, p, pdplans, cstarts, total, Nprof,
                                       nchunks, r_lo, lodr, gpu_device);
            if (!gs) {
                fprintf(stderr, "Error: failed to create the CUDA session\n");
                if (own_plans) { free(cstarts); direct_plans_free(pdplans); }
                free(wpath);
                return empty;
            }
            if (cache) cache->gpu_session = gs;
            else own_session = 1;
        }
        cands = cs_gpu_session_run(gs, ft, threshold, progress_mode);
        if (own_session) cs_gpu_session_free(gs);
    } else {
        Workspace *ws = NULL;
        int own_ws = 0;
        if (cache && cache->ws && cache->Nprof == Nprof && cache->params == p &&
            cache->nthreads >= nt) {
            ws = cache->ws;
        } else {
            if (cache && cache->ws) workspaces_free(cache->ws, cache->nthreads, 0);
            unsigned plan_flags = wisdom ? FFTW_PATIENT : FFTW_MEASURE;
            ws = workspaces_create(p, Nprof, nt, plan_flags);
            if (!ws) { if (own_plans) { free(cstarts); direct_plans_free(pdplans); } free(wpath); return empty; }
            if (cache) {
                cache->ws = ws; cache->Nprof = Nprof; cache->nthreads = nt;
                cache->params = (SearchParams *)p;
            } else own_ws = 1;
        }

        if (wpath) {
            cs_fftw_plan_lock();
            if (p->precision == CS_PREC_F64) fftw_export_wisdom_to_filename(wpath);
            else fftwf_export_wisdom_to_filename(wpath);
            cs_fftw_plan_unlock();
            if (cache) cache->wisdom_done = 1;
        }

        if (p->sigma == CS_SIGMA_ANALYTIC)
            sigma_sanity_check(ft, p, &ws[0], pdplans, r_lo, lodr, total, Nprof);

        MetricNorm *norm = NULL;
        if (normalize) {
            MetricStats *normstats = metricstats ? metricstats : (MetricStats *)malloc(sizeof(MetricStats));
            if (!metricstats) metricstats_init(normstats, 0.0, 64.0, 3200, 16);
            fprintf(stderr, "Info: Normalising: measuring per-(k,frequency) noise (pass 1/2)\n");
            CandidateList dummy = search_region(ft, p, ws, nbins, r_lo, r_hi, lodr,
                                                total, Nprof, nchunks, nt, cstarts,
                                                INFINITY, NULL, normstats, progress_mode,
                                                pdplans);
            candlist_free(&dummy);
            norm = build_metricnorm(normstats);
            fprintf(stderr, "Info: Built in-situ normalisation model; searching (pass 2/2)\n");
            if (!metricstats) { metricstats_free(normstats); free(normstats); }
        }

        cands = search_region(ft, p, ws, nbins, r_lo, r_hi, lodr,
                              total, Nprof, nchunks, nt, cstarts,
                              threshold, norm,
                              normalize ? NULL : metricstats,
                              progress_mode, pdplans);

        metricnorm_free(norm);
        if (own_ws) workspaces_free(ws, nt, p->nharms);
    }

    fprintf(stderr, "Info: Search complete; post-processing candidates (total_above_threshold=%zu)\n",
            cands.n);

    if (remove) {
        size_t n0 = cands.n;
        CandidateList kept = remove_duplicates(&cands, dr_tol);
        fprintf(stderr, "Info: Collapsed near-identical (duplicate) candidates "
                        "(removed=%zu remaining=%zu)\n",
                n0 - kept.n, kept.n);
        candlist_free(&cands);
        cands = kept;
    } else {
        candlist_sort_freq_stable(&cands);
    }
    if (harm_remove) {
        size_t n1 = cands.n;
        CandidateList kept = remove_harmonics(&cands, numharm, harm_tol);
        fprintf(stderr, "Info: Collapsed harmonically-related candidates "
                        "(removed=%zu remaining=%zu)\n",
                n1 - kept.n, kept.n);
        candlist_free(&cands);
        cands = kept;
    }

    if (own_plans) {
        free(cstarts);
        direct_plans_free(pdplans);
    }
    if (wpath) free(wpath);
    return cands;
}

void search_cache_free(SearchCache *cache)
{
    if (!cache) return;
    if (cache->gpu_session) {
        cs_gpu_session_free((GpuSession *)cache->gpu_session);
        cache->gpu_session = NULL;
    }
    if (cache->plan_valid) {
        direct_plans_free(&cache->dplans);
        free(cache->cstarts);
        cache->cstarts = NULL;
        cache->plan_valid = 0;
    }
    if (cache->ws) {
        workspaces_free(cache->ws, cache->nthreads, 0);
        cache->ws = NULL;
    }
    cache->params = NULL;
}
