/* Candidates: lists, collapsing, duty-cycle measurement, output. */
#include "coherent_search.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Candidate list                                                     */
/* ------------------------------------------------------------------ */
void candlist_init(CandidateList *l) { l->v = NULL; l->n = 0; l->cap = 0; }

void candlist_free(CandidateList *l)
{
    free(l->v);
    l->v = NULL; l->n = 0; l->cap = 0;
}

void candlist_push(CandidateList *l, Candidate c)
{
    if (l->n == l->cap) {
        size_t nc = l->cap ? l->cap * 2 : 256;
        l->v = (Candidate *)realloc(l->v, nc * sizeof(Candidate));
        l->cap = nc;
    }
    l->v[l->n++] = c;
}

void candlist_append(CandidateList *dst, const CandidateList *src)
{
    if (src->n == 0) return;
    if (dst->n + src->n > dst->cap) {
        size_t nc = dst->cap ? dst->cap : 256;
        while (nc < dst->n + src->n) nc *= 2;
        dst->v = (Candidate *)realloc(dst->v, nc * sizeof(Candidate));
        dst->cap = nc;
    }
    memcpy(dst->v + dst->n, src->v, src->n * sizeof(Candidate));
    dst->n += src->n;
}

/* ------------------------------------------------------------------ */
/* Stable bottom-up merge sort                                        */
/* ------------------------------------------------------------------ */
typedef int (*cand_cmp_fn)(const Candidate *, const Candidate *);

static void merge_sort(Candidate *v, Candidate *tmp, size_t n, cand_cmp_fn cmp)
{
    if (n < 2) return;
    { size_t width; for (width = 1; width < n; width *= 2) {
        { size_t i; for (i = 0; i < n; i += 2 * width) {
            size_t mid = i + width < n ? i + width : n;
            size_t end = i + 2 * width < n ? i + 2 * width : n;
            size_t a = i, b = mid, k = i;
            while (a < mid && b < end) {
                if (cmp(&v[b], &v[a]) < 0) tmp[k++] = v[b++];
                else tmp[k++] = v[a++];
            }
            while (a < mid) tmp[k++] = v[a++];
            while (b < end) tmp[k++] = v[b++];
        } }
        memcpy(v, tmp, n * sizeof(Candidate));
    } }
}

static int cmp_metric_desc(const Candidate *a, const Candidate *b)
{
    return (b->metric > a->metric) - (b->metric < a->metric);
}
static int cmp_freq_asc(const Candidate *a, const Candidate *b)
{
    return (a->freq > b->freq) - (a->freq < b->freq);
}
static int cmp_r_asc(const Candidate *a, const Candidate *b)
{
    return (a->r > b->r) - (a->r < b->r);
}

static void sort_with(CandidateList *l, cand_cmp_fn cmp)
{
    if (l->n < 2) return;
    Candidate *tmp = (Candidate *)malloc(l->n * sizeof(Candidate));
    if (!tmp) return;
    merge_sort(l->v, tmp, l->n, cmp);
    free(tmp);
}

void candlist_sort_metric_desc(CandidateList *l) { sort_with(l, cmp_metric_desc); }
void candlist_sort_freq(CandidateList *l) { sort_with(l, cmp_freq_asc); }
void candlist_sort_r(CandidateList *l) { sort_with(l, cmp_r_asc); }
void candlist_sort_metric_desc_stable(CandidateList *l) { sort_with(l, cmp_metric_desc); }
void candlist_sort_freq_stable(CandidateList *l) { sort_with(l, cmp_freq_asc); }
void candlist_sort_r_stable(CandidateList *l) { sort_with(l, cmp_r_asc); }

/* ------------------------------------------------------------------ */
/* Candidate collapsing                                               */
/* ------------------------------------------------------------------ */
CandidateList remove_duplicates(const CandidateList *in, double dr_tol)
{
    CandidateList out; candlist_init(&out);
    if (in->n == 0) return out;
    CandidateList tmp; candlist_init(&tmp);
    tmp.v = (Candidate *)malloc(in->n * sizeof(Candidate));
    tmp.n = in->n; tmp.cap = in->n;
    memcpy(tmp.v, in->v, in->n * sizeof(Candidate));
    candlist_sort_r_stable(&tmp);
    Candidate best = tmp.v[0];
    double prev_r = best.r;
    { size_t i; for (i = 1; i < tmp.n; i++) {
        Candidate c = tmp.v[i];
        if (c.r - prev_r <= dr_tol) {
            if (c.metric > best.metric) best = c;
        } else {
            candlist_push(&out, best);
            best = c;
        }
        prev_r = c.r;
    } }
    candlist_push(&out, best);
    candlist_free(&tmp);
    candlist_sort_freq_stable(&out);
    return out;
}

int harmonically_related(double r1, double r2, int numharm, double tol)
{
    double lo = r1 < r2 ? r1 : r2;
    double hi = r1 < r2 ? r2 : r1;
    if (!(lo > 0.0)) return 0;
    { int m; for (m = 1; m <= numharm; m++) {
        int n = (int)nearbyint((double)m * hi / lo);
        if (n < 1 || n > numharm) continue;
        if (fabs((double)m * hi - (double)n * lo) <= tol * (double)m) return 1;
    } }
    return 0;
}

CandidateList remove_harmonics(const CandidateList *in, int numharm, double tol)
{
    CandidateList out; candlist_init(&out);
    if (in->n == 0) return out;
    CandidateList tmp; candlist_init(&tmp);
    tmp.v = (Candidate *)malloc(in->n * sizeof(Candidate));
    tmp.n = in->n; tmp.cap = in->n;
    memcpy(tmp.v, in->v, in->n * sizeof(Candidate));
    candlist_sort_metric_desc_stable(&tmp);
    { size_t i; for (i = 0; i < tmp.n; i++) {
        Candidate c = tmp.v[i];
        int related = 0;
        { size_t j; for (j = 0; j < out.n; j++) {
            if (harmonically_related(c.r, out.v[j].r, numharm, tol)) {
                related = 1;
                break;
            }
        } }
        if (!related) candlist_push(&out, c);
    } }
    candlist_free(&tmp);
    candlist_sort_freq_stable(&out);
    return out;
}

/* ------------------------------------------------------------------ */
/* Per-candidate profile reconstruction                               */
/* ------------------------------------------------------------------ */
/* One FFTW c2r plan per distinct profile length, reused across every
 * candidate of every file in a run.  candidate_profile is called from
 * measure_ducy after the threaded search region and from the plotter, i.e.
 * always sequentially, so a process-global cache is safe here. */
typedef struct {
    int n;
    fftw_plan plan;
    double *in;
    double *out;
} ProfPlan;

#define PROF_PLAN_CACHE 64
static ProfPlan g_prof_plans[PROF_PLAN_CACHE];
static int g_prof_nplans;

static ProfPlan *profile_plan_get(int n)
{
    { int i; for (i = 0; i < g_prof_nplans; i++)
        if (g_prof_plans[i].n == n) return &g_prof_plans[i]; }
    if (g_prof_nplans >= PROF_PLAN_CACHE) return NULL;
    ProfPlan *p = &g_prof_plans[g_prof_nplans];
    p->n = n;
    p->in = (double *)calloc((size_t)(n / 2 + 1) * 2, sizeof(double));
    p->out = (double *)malloc((size_t)n * sizeof(double));
    if (!p->in || !p->out) {
        free(p->in); free(p->out);
        p->in = p->out = NULL;
        return NULL;
    }
    cs_fftw_plan_lock();
    p->plan = fftw_plan_dft_c2r_1d(n, (fftw_complex *)p->in, p->out,
                                   FFTW_ESTIMATE);
    cs_fftw_plan_unlock();
    if (!p->plan) {
        free(p->in); free(p->out);
        p->in = p->out = NULL;
        return NULL;
    }
    g_prof_nplans++;
    return p;
}

double *candidate_profile(const FFTFile *ft, double r, int nharm, int m, int *out_n)
{
    *out_n = 0;
    if (nharm < 1) return NULL;
    const int m2 = m / 2;
    const long long Nhalf = ft->N / 2;
    const long long namps = ft->namps;

    double *amps = (double *)malloc((size_t)nharm * 2 * sizeof(double));
    if (!amps) return NULL;
    int H = 0;
    { int h; for (h = 1; h <= nharm; h++) {
        double rh = r * (double)h;
        long long rint0 = (long long)floor(rh + 1e-15) + 1;
        if (!(rint0 - m2 >= 0 && rint0 + m2 <= namps && rh < (double)Nhalf)) break;
        double re, im;
        fourier_interp_complex(rh, ft->amps, namps, m, &re, &im);
        amps[2 * H] = re;
        amps[2 * H + 1] = im;
        H++;
    } }
    if (H < 1) {
        free(amps);
        double *z = (double *)calloc(2, sizeof(double));
        *out_n = 2;
        return z;
    }
    int n = 2 * H;
    double *out = (double *)malloc((size_t)n * sizeof(double));
    if (!out) { free(amps); return NULL; }
    ProfPlan *pp = profile_plan_get(n);
    if (pp) {
        memset(pp->in, 0, (size_t)(H + 1) * 2 * sizeof(double));
        { int h; for (h = 0; h < H; h++) {
            pp->in[2 * (h + 1)] = amps[2 * h];
            pp->in[2 * (h + 1) + 1] = amps[2 * h + 1];
        } }
        fftw_execute(pp->plan);
        const double inv = 1.0 / (double)n;
        { int i; for (i = 0; i < n; i++) out[i] = pp->out[i] * inv; }
    } else {
        double *stack = (double *)calloc((size_t)(H + 1) * 2, sizeof(double));
        if (!stack) { free(amps); free(out); return NULL; }
        { int h; for (h = 0; h < H; h++) {
            stack[2 * (h + 1)] = amps[2 * h];
            stack[2 * (h + 1) + 1] = amps[2 * h + 1];
        } }
        cs_fftw_plan_lock();
        fftw_plan plan = fftw_plan_dft_c2r_1d(n, (fftw_complex *)stack, out,
                                              FFTW_ESTIMATE);
        cs_fftw_plan_unlock();
        if (!plan) { free(amps); free(stack); free(out); return NULL; }
        fftw_execute(plan);
        cs_fftw_plan_lock();
        fftw_destroy_plan(plan);
        cs_fftw_plan_unlock();
        const double inv = 1.0 / (double)n;
        { int i; for (i = 0; i < n; i++) out[i] *= inv; }
        free(stack);
    }
    free(amps);
    *out_n = n;
    return out;
}

CandidateList read_candidates_file(const char *path, double T)
{
    CandidateList out;
    candlist_init(&out);
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Error: cannot open candidate file '%s'\n", path);
        return out;
    }
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0') continue;
        int rank = 0, nharm = 0;
        double snr = 0.0, freq = 0.0, period = 0.0;
        if (sscanf(p, "%d %lf %lf %lf %d", &rank, &snr, &freq, &period,
                   &nharm) >= 5) {
            if (!(freq > 0.0)) continue;
            Candidate c;
            c.freq = freq;
            c.metric = snr;
            c.r = freq * T;
            c.nharm = nharm;
            c.ducy = NAN;
            candlist_push(&out, c);
        }
    }
    fclose(f);
    return out;
}

CandidateList measure_ducy(const FFTFile *ft, const CandidateList *in,
                           const SearchParams *p)
{
    CandidateList out; candlist_init(&out);
    { size_t i; for (i = 0; i < in->n; i++) {
        Candidate c = in->v[i];
        c.ducy = NAN;
        if (c.nharm >= 1 && c.r > 0.0) {
            int n = 0;
            double *prof = candidate_profile(ft, c.r, c.nharm, 64, &n);
            if (prof) {
                if (n >= 2) {
                    int w = 1;
                    double ducy = NAN;
                    if (boxcar_best_width(prof, n, p->boxcar_fsp,
                                          p->boxcar_maxfrac, &w, &ducy) == 0)
                        c.ducy = ducy;
                }
                free(prof);
            }
        }
        candlist_push(&out, c);
    } }
    return out;
}

/* ------------------------------------------------------------------ */
/* Output                                                             */
/* ------------------------------------------------------------------ */
/* Julia's show(Float64): shortest representation that round-trips, with a
 * trailing ".0" on integral values (so 6.0 prints as "6.0", not "6"). */
void cs_format_double(char *buf, size_t n, double x)
{
    if (isnan(x)) { snprintf(buf, n, "NaN"); return; }
    if (isinf(x)) { snprintf(buf, n, x > 0 ? "Inf" : "-Inf"); return; }
    { int prec; for (prec = 1; prec <= 17; prec++) {
        snprintf(buf, n, "%.*g", prec, x);
        char *end = NULL;
        if (strtod(buf, &end) == x && end && *end == '\0') break;
    } }
    if (!strpbrk(buf, ".eE") && !strstr(buf, "inf") && !strstr(buf, "nan")) {
        size_t l = strlen(buf);
        if (l + 3 < n) { buf[l] = '.'; buf[l + 1] = '0'; buf[l + 2] = '\0'; }
    }
}
int write_candidates(const CandidateList *cands, const char *outfile,
                     double threshold)
{
    FILE *f = NULL;
    int to_stdout = (outfile == NULL || *outfile == '\0');
    if (to_stdout) f = stdout;
    else {
        f = fopen(outfile, "w");
        if (!f) {
            fprintf(stderr, "Error: cannot write candidates to '%s'\n", outfile);
            return -1;
        }
    }
    fprintf(f, "#Num    'S/N'      Frequency (Hz)        Period (ms)    #Harm  Ducy(%%)\n");
    { size_t i; for (i = 0; i < cands->n; i++) {
        const Candidate *c = &cands->v[i];
        char dstr[32];
        if (isnan(c->ducy)) snprintf(dstr, sizeof(dstr), "-");
        else snprintf(dstr, sizeof(dstr), "%.2f", 100.0 * c->ducy);
        fprintf(f, "%-4zu  %8.2f  %18.12f  %18.12f   %3d   %6s\n",
                i + 1, c->metric, c->freq, 1000.0 / c->freq, c->nharm, dstr);
    } }
    if (to_stdout) {
        char tbuf[64];
        cs_format_double(tbuf, sizeof(tbuf), threshold);
        fprintf(stderr, "# %zu candidates above threshold %s\n", cands->n, tbuf);
    } else {
        fclose(f);
        fprintf(stderr, "Info: Wrote candidates (n=%zu file=%s)\n", cands->n, outfile);
    }
    return 0;
}
