#define _GNU_SOURCE
/* Command-line front end and metric-statistics report. */
#include "coherent_search.h"

#include <errno.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* Small utilities                                                    */
/* ------------------------------------------------------------------ */
static char *xstrdup(const char *s)
{
    size_t n = strlen(s) + 1;
    char *r = (char *)malloc(n);
    if (r) memcpy(r, s, n);
    return r;
}

static const char *base_name(const char *path)
{
    const char *s = strrchr(path, '/');
    return s ? s + 1 : path;
}

/* Julia splitext: last dot in the basename, unless it is the first char. */
static char *path_stem(const char *path)
{
    size_t n = strlen(path);
    size_t slash = 0;
    { size_t i; for (i = 0; i < n; i++) if (path[i] == '/') slash = i + 1; }
    size_t dot = n;
    { size_t i; for (i = slash; i < n; i++) if (path[i] == '.') dot = i; }
    if (dot == n || dot == slash) dot = n;
    char *r = (char *)malloc(dot + 1);
    if (!r) return NULL;
    memcpy(r, path, dot);
    r[dot] = '\0';
    return r;
}

static char *base_stem(const char *path)
{
    char *s = path_stem(path);
    if (!s) return NULL;
    char *r = xstrdup(base_name(s));
    free(s);
    return r;
}

static void mkdir_p(const char *path)
{
    if (!path || !*path) return;
    char *p = xstrdup(path);
    if (!p) return;
    { char *q; for (q = p + 1; *q; q++) {
        if (*q == '/') {
            *q = '\0';
            mkdir(p, 0777);
            *q = '/';
        }
    } }
    mkdir(p, 0777);
    free(p);
}

static char *output_path(const char *fftfile, const char *outfilenm,
                         const char *outdir, int nfiles)
{
    if (nfiles == 1 && (!outdir || !*outdir))
        return xstrdup(outfilenm ? outfilenm : "");
    char *stem = path_stem(fftfile);
    if (!stem) return NULL;
    char *name;
    if (outdir && *outdir) {
        const char *bn = base_name(stem);
        size_t n = strlen(outdir) + strlen(bn) + 2;
        name = (char *)malloc(n);
        snprintf(name, n, "%s/%s", outdir, bn);
        free(stem);
    } else {
        name = stem;
    }
    size_t n2 = strlen(name) + 16;
    char *out = (char *)malloc(n2);
    snprintf(out, n2, "%s.cohout", name);
    free(name);
    return out;
}

/* Stem for the profile plots, mirroring Julia's plot_stem(). */
static char *plot_stem_for(const char *plotstem, const char *outfile,
                           const char *fftfile)
{
    if (plotstem && *plotstem) return xstrdup(plotstem);
    if (outfile && *outfile) {
        char *s = xstrdup(outfile);
        size_t l = strlen(s);
        if (l >= 7 && strcmp(s + l - 7, ".cohout") == 0) s[l - 7] = '\0';
        size_t n = strlen(s) + 12;
        char *out = (char *)malloc(n);
        snprintf(out, n, "%s_profiles", s);
        free(s);
        return out;
    }
    char *stem = xstrdup(base_name(fftfile));
    char *dot = strrchr(stem, '.');
    if (dot) *dot = '\0';
    size_t n = strlen(stem) + 12;
    char *out = (char *)malloc(n);
    snprintf(out, n, "%s_profiles", stem);
    free(stem);
    return out;
}

static int cmp_strptr(const void *a, const void *b)
{
    return strcmp(*(char *const *)a, *(char *const *)b);
}

/* Reject multi-input runs whose candidate filenames collide: every input
 * produces <stem>.cohout, so mixing foo.dat with foo.fft (or equal basenames
 * under --outdir) would otherwise silently overwrite one result. */
static int check_output_collisions(char **files, int nfiles, const char *outdir)
{
    if (nfiles <= 1) return 0;
    char **outs = (char **)malloc((size_t)nfiles * sizeof(char *));
    if (!outs) return 0;
    { int i; for (i = 0; i < nfiles; i++)
        outs[i] = output_path(files[i], "", outdir, 2); }
    qsort(outs, (size_t)nfiles, sizeof(char *), cmp_strptr);
    int bad = 0;
    { int i; for (i = 1; i < nfiles; i++) {
        if (outs[i] && outs[i - 1] && strcmp(outs[i], outs[i - 1]) == 0) {
            fprintf(stderr, "Error: several inputs would write the same candidate "
                            "file '%s' (e.g. a .dat and a .fft with the same "
                            "stem, or equal basenames with --outdir)\n", outs[i]);
            bad = 1;
            break;
        }
    } }
    { int i; for (i = 0; i < nfiles; i++) free(outs[i]); }
    free(outs);
    return bad;
}

/* ------------------------------------------------------------------ */
/* Metric statistics report                                           */
/* ------------------------------------------------------------------ */
static void hist_row(const MetricHistogram *h, const double *faps, int nfaps,
                     double *mean, double *std, double *median,
                     double *fapv, double *overflow)
{
    long long N = h->total;
    *mean = N > 0 ? h->sum / (double)N : NAN;
    *std = N > 1 ? sqrt(fmax(0.0, (h->sumsq - (double)N * (*mean) * (*mean)) /
                                  (double)(N - 1))) : 0.0;
    *median = hist_quantile(h, 0.5);
    { int i; for (i = 0; i < nfaps; i++) fapv[i] = hist_quantile(h, 1.0 - faps[i]); }
    *overflow = N > 0 ? (double)h->over / (double)N : 0.0;
}

void report_metricstats(const MetricStats *ms, const char *base,
                        const SearchParams *p, double threshold)
{
    int any = 0;
    { size_t i; for (i = 0; i < ms->nhists; i++) if (ms->hists[i].total > 0) any = 1; }
    if (!any) {
        fprintf(stderr, "Warning: No metric statistics collected (no blocks searched)\n");
        return;
    }
    const double faps[5] = {0.1, 0.01, 1e-3, 1e-4, 1e-5};
    const char *labels[5] = {"1e-1", "1e-2", "1e-3", "1e-4", "1e-5"};

    char thbuf[64];
    cs_format_double(thbuf, sizeof(thbuf), threshold);
    fprintf(stderr, "\nMetric statistics by decimation  (peak boxcar S/N, threshold=%s)\n",
            thbuf);
    fprintf(stderr, "  Pure-noise floor scales ~sqrt(nbins); a fixed threshold picks a DIFFERENT false-alarm rate per k.\n");
    fprintf(stderr, "  %-3s %4s %5s %10s %7s %7s %7s | metric at single-trial FAP =\n",
            "k", "Hk", "nbins", "ntrials", "mean", "std", "max");
    fprintf(stderr, "  %-3s %4s %5s %10s %7s %7s %7s | %8s %8s %8s %8s %8s\n",
            "", "", "", "", "", "", "", labels[0], labels[1], labels[2],
            labels[3], labels[4]);
    { size_t i; for (i = 0; i < ms->nhists; i++) {
        const MetricHistogram *h = &ms->hists[i];
        double mean, std, med, fv[5], ov;
        hist_row(h, faps, 5, &mean, &std, &med, fv, &ov);
        int nblocks = 0;
        { size_t b; for (b = 0; b < ms->nblocks; b++)
            if (ms->blocks[b].k == h->k && ms->blocks[b].n > 0) nblocks++; }
        fprintf(stderr, "  %-3d %4d %5d %10lld %7.3f %7.3f %7.3f |",
                h->k, h->Hk, h->nbins, h->total,
                mean, std, h->total > 0 ? h->vmax : NAN);
        { int q; for (q = 0; q < 5; q++) fprintf(stderr, " %8.3f", fv[q]); }
        if (ov > faps[4]) fprintf(stderr, "  (overflow %.1e!)", ov);
        fprintf(stderr, "\n");
    } }
    fprintf(stderr, "  A single --threshold gives these per-k FAPs; for comparable FAP, threshold each k separately\n");
    fprintf(stderr, "  (or normalise the metric per (k, frequency)).  FAP columns are single-trial, single-decimation.\n");

    fprintf(stderr, "\n  FAP=1e-4 threshold vs frequency (%d log-spaced windows per k; drift = red noise / Nyquist):\n",
            ms->nwin);
    fprintf(stderr, "  %-3s %5s %21s %9s %9s\n", "k", "nbins", "freq range (Hz)", "min", "max");
    int kseen[256]; int nk = 0;
    { size_t i; for (i = 0; i < ms->nwhists; i++) {
        int k = ms->whists[i].k, dup = 0;
        { int j; for (j = 0; j < nk; j++) if (kseen[j] == k) dup = 1; }
        if (!dup && nk < 256) kseen[nk++] = k;
    } }
    { int j; for (j = 0; j < nk; j++) {
        int k = kseen[j];
        double vmin = INFINITY, vmax = -INFINITY, fl = INFINITY, fh = -INFINITY;
        int nbins = 0, first = 1;
        { size_t i; for (i = 0; i < ms->nwhists; i++) {
            const MetricHistogram *h = &ms->whists[i];
            if (h->k != k) continue;
            double fv = hist_quantile(h, 1.0 - 1e-4);
            if (fv < vmin) vmin = fv;
            if (fv > vmax) vmax = fv;
            if (h->flo < fl) fl = h->flo;
            if (h->fhi > fh) fh = h->fhi;
            if (first) { nbins = h->nbins; first = 0; }
        } }
        fprintf(stderr, "  %-3d %5d %9.3f -%9.3f %9.3f %9.3f\n",
                k, nbins, fl, fh, vmin, vmax);
    } }
    fprintf(stderr, "  Full per-window thresholds -> %s_metricfap.txt\n\n", base);

    char fname[4096];
    snprintf(fname, sizeof(fname), "%s_metricstats.txt", base);
    FILE *io = fopen(fname, "w");
    if (io) {
        fprintf(io, "# Per-block, per-decimation metric statistics\n");
        fprintf(io, "# metric=boxcar nharms=%d threshold=%s\n", p->nharms, thbuf);
        fprintf(io, "# nbins = 2*Hk; the pure-noise metric floor scales ~sqrt(nbins).\n");
        fprintf(io, "#%-7s %3s %4s %5s %10s %14s %14s %8s %9s %9s %9s %9s %9s\n",
                "block", "k", "Hk", "nbins", "ngoodbins", "f_lo(Hz)", "f_hi(Hz)",
                "n", "min", "median", "mean", "std", "max");
        { size_t i; for (i = 0; i < ms->nblocks; i++) {
            const BlockMetricStats *s = &ms->blocks[i];
            fprintf(io, "%-8d %3d %4d %5d %10.3f %14.8f %14.8f %8lld %9.3f %9.3f %9.3f %9.3f %9.3f\n",
                    s->block, s->k, s->Hk, s->nbins, s->ngoodbins, s->flo, s->fhi,
                    s->n, s->min, s->median, s->mean, s->std, s->max);
        } }
        fclose(io);
    }

    snprintf(fname, sizeof(fname), "%s_metricfap.txt", base);
    io = fopen(fname, "w");
    if (io) {
        fprintf(io, "# Per-(decimation, frequency-window) empirical false-alarm thresholds\n");
        fprintf(io, "# metric=boxcar nharms=%d\n", p->nharms);
        fprintf(io, "# fap_X = metric value whose single-trial, single-decimation false-alarm prob is X (in-window)\n");
        fprintf(io, "#%-3s %4s %5s %4s %13s %13s %10s %8s %8s %8s | %s\n",
                "k", "Hk", "nbins", "win", "f_lo(Hz)", "f_hi(Hz)", "ntrials",
                "mean", "std", "median", "fap1e-1 fap1e-2 fap1e-3 fap1e-4 fap1e-5");
        { size_t i; for (i = 0; i < ms->nwhists; i++) {
            const MetricHistogram *r = &ms->whists[i];
            double mean, std, med, fv[5], ov;
            hist_row(r, faps, 5, &mean, &std, &med, fv, &ov);
            fprintf(io, "%-4d %4d %5d %4d %13.7f %13.7f %10lld %8.3f %8.3f %8.3f |",
                    r->k, r->Hk, r->nbins, r->win, r->flo, r->fhi, r->total,
                    mean, std, med);
            { int q; for (q = 0; q < 5; q++) fprintf(io, " %7.3f", fv[q]); }
            if (ov > faps[4]) fprintf(io, "  # overflow %.1e", ov);
            fprintf(io, "\n");
        } }
        fclose(io);
    }

    snprintf(fname, sizeof(fname), "%s_metrichist.txt", base);
    io = fopen(fname, "w");
    if (io) {
        fprintf(io, "# Per-(decimation, frequency-window) metric histograms (one streaming pass)\n");
        fprintf(io, "# metric=boxcar nharms=%d\n", p->nharms);
        fprintf(io, "# bin = left edge of a bin of width %g; count = trials in [bin, bin+width)\n",
                (ms->hist_hi - ms->hist_lo) / (double)ms->hist_nb);
        fprintf(io, "#%-3s %4s %4s %13s %13s %12s %14s\n",
                "k", "Hk", "win", "f_lo(Hz)", "f_hi(Hz)", "bin", "count");
        double w = (ms->hist_hi - ms->hist_lo) / (double)ms->hist_nb;
        { size_t i; for (i = 0; i < ms->nwhists; i++) {
            const MetricHistogram *h = &ms->whists[i];
            int firstnz = -1, lastnz = -1;
            { int b; for (b = 0; b < h->nb; b++) {
                if (h->counts[b] > 0) { if (firstnz < 0) firstnz = b; lastnz = b; }
            } }
            if (firstnz < 0) continue;
            { int b; for (b = firstnz; b <= lastnz; b++)
                fprintf(io, "%-4d %4d %4d %13.7f %13.7f %12.5f %14lld\n",
                        h->k, h->Hk, h->win, h->flo, h->fhi,
                        ms->hist_lo + (double)b * w, h->counts[b]); }
        } }
        fclose(io);
    }
    fprintf(stderr, "Info: Wrote metric statistics (perblock=%s_metricstats.txt fap=%s_metricfap.txt histogram=%s_metrichist.txt nblocks=%zu)\n",
            base, base, base, ms->nblocks);
}

/* ------------------------------------------------------------------ */
/* Options                                                            */
/* ------------------------------------------------------------------ */
typedef struct {
    char **files;
    int nfiles;
    double threshold;
    char *outfilenm;
    char *outdir;
    int nharms;
    int ncands;
    int lobin;
    double lofreq;
    double hifreq;
    double hidr;
    int m;
    cs_precision precision;
    cs_sigma sigma;
    int blocksize;
    int maxdecim;
    double drtol;
    int numharm;
    int remove;
    int harm_remove;
    int progressbar;
    int noprogress;
    int verbose;
    int metricstats;
    int normalize;
    int nowisdom;
    char *wisdomfile;
    int plot;
    int gpu;
    int gpu_device;
    char *plotstem;
    int plotcols;
    int plotrows;
    int files_owned;
    int threads;
    int prime_wisdom;
    int nharms_set;
    char *plot_candidates;
    char *zaplist;
    double baryv;
} Options;

static void opts_defaults(Options *o)
{
    memset(o, 0, sizeof(*o));
    o->threshold = 8.0;
    o->outfilenm = xstrdup("");
    o->outdir = xstrdup("");
    o->nharms = 60;
    o->ncands = 100;
    o->lobin = 100;
    o->lofreq = 0.1;
    o->hifreq = 125.0;
    o->hidr = 0.5;
    o->m = 16;
    o->precision = CS_PREC_F32;
    o->sigma = CS_SIGMA_ANALYTIC;
    o->blocksize = 0;
    o->maxdecim = 6;
    o->drtol = 1.0;
    o->numharm = 16;
    o->remove = 1;
    o->harm_remove = 1;
    o->wisdomfile = xstrdup("");
    o->plotstem = xstrdup("");
    o->plot_candidates = xstrdup("");
    o->zaplist = xstrdup("");
    o->baryv = 0.0;
    o->plotcols = 3;
    o->plotrows = 5;
    o->gpu_device = 0;
    o->threads = 0;
}

void cs_usage_mode(FILE *f, int list_mode, int show_examples)
{
    if (list_mode)
        fprintf(f, "Usage: coherent_search_list [options] FILE.dat.list\n");
    else
        fprintf(f, "Usage: coherent_search [options] FILE.fft|FILE.dat [FILE2 ...]\n");
    fprintf(f,
"\n"
"Search PRESTO-style Fourier transforms for pulsars using coherent harmonic\n"
"summing.  Raw .dat time series are accepted too: PRESTO's realfft and rednoise\n"
"steps are performed in memory exactly as `realfft name.dat && rednoise name.fft`\n"
"would (default startwidth=6, endwidth=100, endfreq=6 Hz), and no .fft or _red.inf\n"
"file is written.\n"
"\n"
"Inputs:\n"
"  FILE.fft        PRESTO Fourier transform (should be normalised; a rednoise\n"
"                  pass is strongly recommended).\n"
"  FILE.dat        single-precision time series with a matching FILE.inf.\n"
"  FILE.dat.list   list mode only: one .dat or .fft path per line, relative to\n"
"                  the list file's directory; blank lines and # comments skipped.\n"
"\n"
"Output:\n"
"  one input, no -o      candidates go to stdout\n"
"  -o FILE               candidates go to FILE\n"
"  several inputs or --outdir\n"
"                        <stem>.cohout per input, i.e. the input suffix is\n"
"                        replaced by .cohout (.dat and .fft alike)\n"
"\n"
"Options:\n"
"  -t, --threshold X       S/N cutoff for picking candidates (default 8.0)\n"
"  -o, --outputfilenm F    output candidate file (single input; default stdout)\n"
"      --outdir D          write per-file .cohout candidate files into D\n"
"  -n, --nharms N          number of harmonics to sum (default 60)\n"
"      --ncands N          maximum number of candidates to report (default 100)\n"
"      --lobin N           lowest frequency bin to search (default 100)\n"
"      --lofreq F          lowest frequency in Hz (default 0.1)\n"
"      --hifreq F          highest FUNDAMENTAL frequency in Hz (default 125)\n"
"      --hidr F            Fourier-bin step at the highest harmonic (default 0.5)\n"
"      --m N               Fourier bins in the interpolation kernel, even (default 16)\n"
"      --precision P       'f32' (default) or 'f64' profile stage\n"
"      --sigma S           'analytic' (default) or 'measured' noise scale\n"
"      --blocksize N       trial fundamentals per chunk\n"
"                          (0 => 2048 on CPU, 65536 on GPU)\n"
"      --maxdecim N        max harmonic-decimation factor (default 6)\n"
"      --drtol F           Fourier-bin tolerance for duplicate collapsing (default 1)\n"
"      --numharm N         max harmonic number for harmonic collapsing (default 16)\n"
"      --noremove          do not collapse near-identical candidates\n"
"      --noharmremove      do not collapse harmonically-related candidates\n"
"      --threads N         OpenMP threads per process.  Default: OMP_NUM_THREADS,\n"
"                          otherwise 1 (one process per input is the intended\n"
"                          multi-core deployment; use --threads N or\n"
"                          OMP_NUM_THREADS for a faster single run)\n"
"      --progressbar       show a progress bar instead of the text percentage\n"
"      --noprogress        do not print a progress meter\n"
"  -v, --verbose           report the trial grid and interpolation plan\n"
"      --metricstats       write per-block/per-decimation metric statistics\n"
"                          (CPU only; not available with --gpu)\n"
"      --normalize         two-pass adaptive threshold, metric in noise-sigma\n"
"                          units (CPU only; doubles the runtime)\n"
"      --nowisdom          do not import/export the FFTW wisdom cache\n"
"      --wisdomfile F      FFTW wisdom cache path\n"
"      --prime-wisdom      build and save all FFTW plans for the current\n"
"                          configuration (f32 and f64), then exit\n"
"      --gpu [N]           run on CUDA device N (default 0; device numbering\n"
"                          starts at 0, as in nvidia-smi).  Requires a GPU=1 build.\n"
"                          GPU runs require --sigma analytic and do not support\n"
"                          --normalize or --metricstats.  A non-rational --hidr\n"
"                          falls back to the CPU.\n"
"      --plot              plot candidate pulse profiles after searching\n"
"                          (SVG pages named <stem>_profiles_NN.svg)\n"
"      --plotstem F        output stem for the profile plots\n"
"      --plotcols N        profile-plot grid columns per page (default 3)\n"
"      --plotrows N        profile-plot grid rows per page (default 5)\n"
"      --plot-candidates F  re-plot from a saved candidate file instead of\n"
"                          searching (one input file; profiles are folded at\n"
"                          the candidate #Harm unless -n is given)\n"
"      --zaplist F         zap frequencies listed in F (two columns: Hz and\n"
"                          width in Hz; lines starting with B are barycentric)\n"
"      --baryv V           radial velocity v/c towards the target, applied to\n"
"                          non-B zaplist entries (default 0.0)\n"
"      --noplot            accepted and ignored (plotting is off by default)\n"
"      --help              show this help\n");
    if (show_examples) {
        fprintf(f,
"\n"
"Python tools (in python/):\n"
"  sift_candidates.py    cross-observation/DM sifting of .cohout files\n"
"  parallel_search.py    split a file list over CPU processes or CUDA devices\n"
"\n"
"Examples:\n"
"  coherent_search -t 5.5 --maxdecim 16 obs.dat\n"
"  coherent_search -t 8 --outdir cands *.fft\n"
"  OMP_NUM_THREADS=8 coherent_search_list -t 5.5 --outdir cands dms.dat.list\n"
"  coherent_search --gpu 0 -t 8 --blocksize 65536 obs.dat\n");
    }
}

void cs_usage(FILE *f, int list_mode)
{
    cs_usage_mode(f, list_mode, 1);
}

static int parse_double(const char *s, double *v)
{
    char *end = NULL;
    errno = 0;
    double x = strtod(s, &end);
    if (errno || !end || *end) return -1;
    *v = x;
    return 0;
}
static int parse_int(const char *s, int *v)
{
    char *end = NULL;
    errno = 0;
    long x = strtol(s, &end, 10);
    if (errno || !end || *end) return -1;
    *v = (int)x;
    return 0;
}

static char *trim_line(char *s)
{
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
    char *e = s + strlen(s);
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r' || e[-1] == '\n'))
        *--e = '\0';
    return s;
}

/* Read a .dat.list: one path per line, relative to the list file's directory. */
static char **read_list(const char *listpath, int *nout)
{
    FILE *f = fopen(listpath, "r");
    if (!f) {
        fprintf(stderr, "Error: cannot open list file '%s'\n", listpath);
        return NULL;
    }
    char *dir = xstrdup(listpath);
    char *slash = strrchr(dir, '/');
    if (slash) *slash = '\0';
    else { free(dir); dir = xstrdup(""); }

    char **v = NULL;
    int n = 0, cap = 0;
    char *line = NULL;
    size_t lcap = 0;
    ssize_t len;
    while ((len = getline(&line, &lcap, f)) >= 0) {
        (void)len;
        char *p = trim_line(line);
        if (!*p || *p == '#') continue;
        char *full;
        if (p[0] == '/' || dir[0] == '\0') {
            full = xstrdup(p);
        } else {
            size_t need = strlen(dir) + strlen(p) + 2;
            full = (char *)malloc(need);
            snprintf(full, need, "%s/%s", dir, p);
        }
        if (n == cap) {
            cap = cap ? cap * 2 : 64;
            v = (char **)realloc(v, (size_t)cap * sizeof(char *));
        }
        v[n++] = full;
    }
    free(line);
    free(dir);
    fclose(f);
    *nout = n;
    return v;
}

/* ------------------------------------------------------------------ */
/* Input prefetch pipeline                                            */
/*                                                                    */
/* A few background threads read and prepare upcoming files (including  */
/* the in-memory realfft/rednoise for .dat) while the main thread        */
/* searches the current one.  With two producers and four slots the      */
/* ~0.05 s/file preparation is fully hidden behind even a GPU search.    */
/* ------------------------------------------------------------------ */
#define PIPE_SLOTS 4
#define PIPE_PROD 2

typedef struct {
    char **paths;
    int n;
    int nslots;
    int nprod;
    int abort;
    DatPipe *pipes[PIPE_SLOTS];
    FFTFile ft[PIPE_SLOTS];
    int ready[PIPE_SLOTS];
    int ok[PIPE_SLOTS];
    long long slot_index[PIPE_SLOTS];
    int consumed;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_t threads[PIPE_PROD];
    struct ProducerArg *pargs;
} InputPipeline;

typedef struct ProducerArg {
    InputPipeline *p;
    int tid;
} ProducerArg;

static void *pipeline_producer(void *arg)
{
    ProducerArg *pa = (ProducerArg *)arg;
    InputPipeline *p = pa->p;
    { int i; for (i = pa->tid; i < p->n; i += p->nprod) {
        pthread_mutex_lock(&p->mu);
        while (!p->abort && i - p->consumed >= p->nslots)
            pthread_cond_wait(&p->cv, &p->mu);
        if (p->abort) {
            pthread_mutex_unlock(&p->mu);
            break;
        }
        int slot = i % p->nslots;
        pthread_mutex_unlock(&p->mu);

        int is_dat = 0;
        int ok = (cs_open_input(p->paths[i], p->pipes[slot],
                                &p->ft[slot], &is_dat) == 0);

        pthread_mutex_lock(&p->mu);
        p->ok[slot] = ok;
        p->slot_index[slot] = i;
        p->ready[slot] = 1;
        pthread_cond_broadcast(&p->cv);
        pthread_mutex_unlock(&p->mu);
    } }
    return NULL;
}

static int pipeline_init(InputPipeline *p, char **paths, int n)
{
    memset(p, 0, sizeof(*p));
    p->paths = paths;
    p->n = n;
    p->nslots = n < PIPE_SLOTS ? n : PIPE_SLOTS;
    p->nprod = n < PIPE_PROD ? n : PIPE_PROD;
    if (p->nprod > p->nslots) p->nprod = p->nslots;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv, NULL);
    { int i; for (i = 0; i < p->nslots; i++) {
        p->pipes[i] = datpipe_create(6, 100, 6.0);
        if (!p->pipes[i]) goto fail;
    } }
    p->pargs = (struct ProducerArg *)calloc((size_t)p->nprod,
                                            sizeof(struct ProducerArg));
    if (!p->pargs) goto fail;
    { int t; for (t = 0; t < p->nprod; t++) {
        p->pargs[t].p = p;
        p->pargs[t].tid = t;
        if (pthread_create(&p->threads[t], NULL, pipeline_producer,
                           &p->pargs[t]) != 0) {
            p->abort = 1;
            pthread_mutex_lock(&p->mu);
            pthread_cond_broadcast(&p->cv);
            pthread_mutex_unlock(&p->mu);
            { int j; for (j = 0; j < t; j++) pthread_join(p->threads[j], NULL); }
            free(p->pargs);
            p->pargs = NULL;
            goto fail;
        }
    } }
    return 0;
fail:
    { int i; for (i = 0; i < p->nslots; i++) datpipe_free(p->pipes[i]); }
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv);
    return -1;
}

static void pipeline_get(InputPipeline *p, int i, FFTFile *out, int *ok)
{
    pthread_mutex_lock(&p->mu);
    int slot = i % p->nslots;
    while (!p->ready[slot] || p->slot_index[slot] != i)
        pthread_cond_wait(&p->cv, &p->mu);
    *out = p->ft[slot];
    *ok = p->ok[slot];
    p->ready[slot] = 0;
    pthread_mutex_unlock(&p->mu);
}

static void pipeline_done(InputPipeline *p)
{
    pthread_mutex_lock(&p->mu);
    p->consumed++;
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
}

static void pipeline_free(InputPipeline *p)
{
    { int t; for (t = 0; t < p->nprod; t++) pthread_join(p->threads[t], NULL); }
    free(p->pargs);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv);
    { int i; for (i = 0; i < p->nslots; i++) datpipe_free(p->pipes[i]); }
}

static int cs_main_mode(int argc, char **argv, int list_mode)
{
    Options o;
    opts_defaults(&o);
    o.files = (char **)calloc((size_t)argc, sizeof(char *));

    { int i; for (i = 1; i < argc; i++) {
        char *arg = argv[i];
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            cs_usage(stdout, list_mode);
            free(o.files);
            return 0;
        }
        if (arg[0] != '-' || strcmp(arg, "-") == 0) {
            o.files[o.nfiles++] = arg;
            continue;
        }
        char *name = arg;
        char *val = NULL;
        if (strncmp(arg, "--", 2) == 0) {
            char *eq = strchr(arg, '=');
            if (eq) { *eq = '\0'; val = eq + 1; }
        }
        #define NEEDVAL() do { \
            if (!val) { if (i + 1 >= argc) { fprintf(stderr, "Error: %s needs a value\n", name); free(o.files); return 2; } val = argv[++i]; } \
        } while (0)

        if (!strcmp(name, "-t") || !strcmp(name, "--threshold")) {
            NEEDVAL();
            if (parse_double(val, &o.threshold)) { fprintf(stderr, "Error: bad threshold '%s'\n", val); return 2; }
        } else if (!strcmp(name, "-o") || !strcmp(name, "--outputfilenm")) {
            NEEDVAL(); free(o.outfilenm); o.outfilenm = xstrdup(val);
        } else if (!strcmp(name, "--outdir")) {
            NEEDVAL(); free(o.outdir); o.outdir = xstrdup(val);
        } else if (!strcmp(name, "-n") || !strcmp(name, "--nharms")) {
            NEEDVAL();
            if (parse_int(val, &o.nharms)) { fprintf(stderr, "Error: bad nharms '%s'\n", val); return 2; }
            o.nharms_set = 1;
        } else if (!strcmp(name, "--ncands")) {
            NEEDVAL();
            if (parse_int(val, &o.ncands)) { fprintf(stderr, "Error: bad ncands '%s'\n", val); return 2; }
        } else if (!strcmp(name, "--lobin")) {
            NEEDVAL();
            if (parse_int(val, &o.lobin)) { fprintf(stderr, "Error: bad lobin '%s'\n", val); return 2; }
        } else if (!strcmp(name, "--lofreq")) {
            NEEDVAL();
            if (parse_double(val, &o.lofreq)) { fprintf(stderr, "Error: bad lofreq '%s'\n", val); return 2; }
        } else if (!strcmp(name, "--hifreq")) {
            NEEDVAL();
            if (parse_double(val, &o.hifreq)) { fprintf(stderr, "Error: bad hifreq '%s'\n", val); return 2; }
        } else if (!strcmp(name, "--hidr")) {
            NEEDVAL();
            if (parse_double(val, &o.hidr)) { fprintf(stderr, "Error: bad hidr '%s'\n", val); return 2; }
        } else if (!strcmp(name, "--m")) {
            NEEDVAL();
            if (parse_int(val, &o.m)) { fprintf(stderr, "Error: bad m '%s'\n", val); return 2; }
        } else if (!strcmp(name, "--precision")) {
            NEEDVAL();
            if (!strcmp(val, "f32")) o.precision = CS_PREC_F32;
            else if (!strcmp(val, "f64")) o.precision = CS_PREC_F64;
            else { fprintf(stderr, "Error: precision must be f32 or f64\n"); return 2; }
        } else if (!strcmp(name, "--sigma")) {
            NEEDVAL();
            if (!strcmp(val, "analytic")) o.sigma = CS_SIGMA_ANALYTIC;
            else if (!strcmp(val, "measured")) o.sigma = CS_SIGMA_MEASURED;
            else { fprintf(stderr, "Error: sigma must be analytic or measured\n"); return 2; }
        } else if (!strcmp(name, "--blocksize")) {
            NEEDVAL();
            if (parse_int(val, &o.blocksize)) { fprintf(stderr, "Error: bad blocksize '%s'\n", val); return 2; }
        } else if (!strcmp(name, "--maxdecim")) {
            NEEDVAL();
            if (parse_int(val, &o.maxdecim)) { fprintf(stderr, "Error: bad maxdecim '%s'\n", val); return 2; }
        } else if (!strcmp(name, "--drtol")) {
            NEEDVAL();
            if (parse_double(val, &o.drtol)) { fprintf(stderr, "Error: bad drtol '%s'\n", val); return 2; }
        } else if (!strcmp(name, "--numharm")) {
            NEEDVAL();
            if (parse_int(val, &o.numharm)) { fprintf(stderr, "Error: bad numharm '%s'\n", val); return 2; }
        } else if (!strcmp(name, "--wisdomfile")) {
            NEEDVAL(); free(o.wisdomfile); o.wisdomfile = xstrdup(val);
        } else if (!strcmp(name, "--plotstem")) {
            NEEDVAL(); free(o.plotstem); o.plotstem = xstrdup(val);
        } else if (!strcmp(name, "--plot-candidates")) {
            NEEDVAL(); free(o.plot_candidates); o.plot_candidates = xstrdup(val);
        } else if (!strcmp(name, "--zaplist")) {
            NEEDVAL(); free(o.zaplist); o.zaplist = xstrdup(val);
        } else if (!strcmp(name, "--baryv")) {
            NEEDVAL();
            if (parse_double(val, &o.baryv)) {
                fprintf(stderr, "Error: bad baryv '%s'\n", val);
                free(o.files);
                return 2;
            }
        } else if (!strcmp(name, "--plotcols")) {
            NEEDVAL();
            if (parse_int(val, &o.plotcols)) { fprintf(stderr, "Error: bad plotcols '%s'\n", val); return 2; }
        } else if (!strcmp(name, "--plotrows")) {
            NEEDVAL();
            if (parse_int(val, &o.plotrows)) { fprintf(stderr, "Error: bad plotrows '%s'\n", val); return 2; }
        } else if (!strcmp(name, "--noremove")) o.remove = 0;
        else if (!strcmp(name, "--noharmremove")) o.harm_remove = 0;
        else if (!strcmp(name, "--progressbar")) o.progressbar = 1;
        else if (!strcmp(name, "--noprogress")) o.noprogress = 1;
        else if (!strcmp(name, "-v") || !strcmp(name, "--verbose")) o.verbose = 1;
        else if (!strcmp(name, "--metricstats")) o.metricstats = 1;
        else if (!strcmp(name, "--normalize")) o.normalize = 1;
        else if (!strcmp(name, "--threads")) {
            NEEDVAL();
            if (parse_int(val, &o.threads) || o.threads < 1) {
                fprintf(stderr, "Error: --threads expects a positive integer\n");
                free(o.files);
                return 2;
            }
        }
        else if (!strcmp(name, "--prime-wisdom")) o.prime_wisdom = 1;
        else if (!strcmp(name, "--nowisdom")) o.nowisdom = 1;
        else if (!strcmp(name, "--plot")) o.plot = 1;
        else if (!strcmp(name, "--gpu")) {
            o.gpu = 1;
            if (val) {
                if (parse_int(val, &o.gpu_device) || o.gpu_device < 0) {
                    fprintf(stderr, "Error: --gpu expects a non-negative device index\n");
                    free(o.files);
                    return 2;
                }
            } else if (i + 1 < argc) {
                /* `--gpu a` selects device a; `--gpu FILE.fft` leaves the file
                 * positional because it is not a bare integer. */
                char *end = NULL;
                long v = strtol(argv[i + 1], &end, 10);
                if (end && end != argv[i + 1] && *end == '\0' && v >= 0) {
                    o.gpu_device = (int)v;
                    i++;
                }
            }
        }
        else if (!strcmp(name, "--noplot")) { /* accepted and ignored */ }
        else {
            fprintf(stderr, "Error: unknown option '%s'\n", name);
            cs_usage(stderr, list_mode);
            free(o.files);
            return 2;
        }
        #undef NEEDVAL
    } }

    if (o.plot_candidates[0]) {
        if (list_mode) {
            fprintf(stderr, "Error: --plot-candidates is for a single input\n");
            free(o.files);
            return 2;
        }
        if (o.nfiles != 1) {
            fprintf(stderr, "Error: --plot-candidates needs exactly one input file\n");
            free(o.files);
            return 2;
        }
    }

    if (list_mode) {
        if (o.nfiles != 1) {
            fprintf(stderr, "Error: coherent_search_list expects exactly one .list file\n");
            cs_usage(stderr, list_mode);
            free(o.files);
            return 2;
        }
        if (o.outfilenm[0]) {
            fprintf(stderr, "Error: -o is not allowed with a list (use --outdir)\n");
            free(o.files);
            return 2;
        }
        int ln = 0;
        char **lv = read_list(o.files[0], &ln);
        free(o.files);
        o.files = lv;
        o.nfiles = ln;
        o.files_owned = 1;
        if (!lv || ln == 0) {
            fprintf(stderr, "Error: list file '%s' contains no inputs\n",
                    argv[argc - 1]);
            return 2;
        }
    } else if (o.nfiles == 0 && !o.prime_wisdom) {
        fprintf(stderr, "Error: at least one FFT file is required\n");
        cs_usage(stderr, list_mode);
        free(o.files);
        return 2;
    }
    /* CPU default is one thread: the intended deployment is one process per
     * .dat/.fft across all cores.  --threads or OMP_NUM_THREADS opt into a
     * multi-threaded process (e.g. --threads 14 for a single fast run). */
    if (o.threads > 0) omp_set_num_threads(o.threads);
    else if (!getenv("OMP_NUM_THREADS")) omp_set_num_threads(1);

    if (o.nfiles > 1 && o.outfilenm[0]) {
        fprintf(stderr, "Error: -o/--outputfilenm names one output file but %d FFT files "
                        "were given; omit it or use --outdir\n", o.nfiles);
        free(o.files);
        return 2;
    }
    if (o.nfiles > 1 && o.plotstem[0]) {
        fprintf(stderr, "Error: --plotstem names one plot stem but %d FFT files were given\n",
                o.nfiles);
        free(o.files);
        return 2;
    }
    if (o.m <= 0 || (o.m & 1)) {
        fprintf(stderr, "Error: m must be a positive even integer\n");
        free(o.files);
        return 2;
    }
    if (o.nharms < 1) {
        fprintf(stderr, "Error: nharms must be positive\n");
        free(o.files);
        return 2;
    }
    int use_gpu = -1;
    if (o.gpu) {
        if (o.sigma != CS_SIGMA_ANALYTIC) {
            fprintf(stderr, "Error: --gpu requires --sigma analytic (the default)\n");
            free(o.files);
            return 2;
        }
        if (o.normalize || o.metricstats) {
            fprintf(stderr, "Error: --gpu does not support --normalize or --metricstats\n");
            free(o.files);
            return 2;
        }
        char gerr[256] = {0};
        int g = cs_gpu_available(o.gpu_device, gerr, sizeof(gerr));
        if (g == 1) {
            use_gpu = o.gpu_device;
            if (o.blocksize == 0) o.blocksize = 65536;
            fprintf(stderr, "Info: using CUDA device %d\n", use_gpu);
        } else if (g == 0) {
            fprintf(stderr, "Warning: %s; running on the CPU (results are equivalent "
                            "to the GPU path to ~2e-7)\n", gerr);
        } else {
            fprintf(stderr, "Error: %s\n", gerr);
            free(o.files);
            return 2;
        }
    }
    if (o.plotstem[0] && !o.plot)
        fprintf(stderr, "Warning: --plotstem given but plotting is off\n");

    if (check_output_collisions(o.files, o.nfiles, o.outdir)) {
        { int i; for (i = 0; i < o.nfiles; i++) if (o.files_owned) free(o.files[i]); }
        free(o.files);
        free(o.outfilenm); free(o.outdir); free(o.wisdomfile);
        free(o.plotstem); free(o.plot_candidates); free(o.zaplist);
        return 2;
    }

    if (o.outdir[0]) mkdir_p(o.outdir);

    SearchParams params;
    params_defaults(&params);
    params.nharms = o.nharms;
    params.m = o.m;
    params.hidr = o.hidr;
    params.threshold = o.threshold;
    params.precision = o.precision;
    params.sigma = o.sigma;
    params.ndecim = decimation_set(o.nharms, o.maxdecim, params.decimations, 64);

    if (o.prime_wisdom) {
        int nprof = o.blocksize > 0 ? o.blocksize : CS_CPU_DEFAULT_BLOCKSIZE;
        int rc = cs_prime_wisdom(&params, nprof, !o.nowisdom, o.wisdomfile);
        free(o.outfilenm); free(o.outdir); free(o.wisdomfile);
        free(o.plotstem); free(o.plot_candidates); free(o.zaplist); free(o.files);
        return rc ? 1 : 0;
    }

    ZapList zaplist;
    int have_zap = 0;
    if (o.zaplist[0]) {
        if (zaplist_read(o.zaplist, &zaplist) != 0) {
            fprintf(stderr, "Error: cannot read zaplist '%s'\n", o.zaplist);
            free(o.outfilenm); free(o.outdir); free(o.wisdomfile);
            free(o.plotstem); free(o.plot_candidates); free(o.zaplist);
            free(o.files);
            return 2;
        }
        have_zap = 1;
        fprintf(stderr, "Info: Read %d birdies from '%s'\n",
                zaplist.n, o.zaplist);
    } else {
        memset(&zaplist, 0, sizeof(zaplist));
    }

    int progress_mode = o.noprogress ? 0 : (o.progressbar ? 2 : 1);

    SearchCache cache;
    memset(&cache, 0, sizeof(cache));
    cache.params = NULL;
    DatPipe *pipe = NULL;
    InputPipeline pipeline;
    int use_pipeline = (o.nfiles > 1);
    /* A default single-threaded CPU process must stay at one core; with GPU
     * the host is idle during kernels, so the prefetch threads are free. */
    if (use_pipeline && use_gpu < 0 && omp_get_max_threads() <= 1)
        use_pipeline = 0;
    if (use_pipeline) {
        if (pipeline_init(&pipeline, o.files, o.nfiles) != 0) {
            fprintf(stderr, "Warning: cannot start the prefetch threads; "
                            "running sequentially\n");
            use_pipeline = 0;
        }
    }
    if (!use_pipeline) pipe = datpipe_create(6, 100, 6.0);

    int exit_status = 0;
    { int fi; for (fi = 0; fi < o.nfiles; fi++) {
        const char *path = o.files[fi];
        if (o.nfiles > 1)
            fprintf(stderr, "Info: File %d of %d (file=%s)\n", fi + 1, o.nfiles, path);
        FFTFile ft;
        int is_dat = 0;
        if (use_pipeline) {
            int prep_ok = 0;
            pipeline_get(&pipeline, fi, &ft, &prep_ok);
            if (!prep_ok) {
                fprintf(stderr, "Error: failed to prepare '%s'\n", path);
                exit_status = 1;
                pipeline_done(&pipeline);
                continue;
            }
        } else if (cs_open_input(path, pipe, &ft, &is_dat) != 0) {
            fprintf(stderr, "Error: failed to open '%s'\n", path);
            exit_status = 1;
            continue;
        }

        if (have_zap) {
            int nz = zaplist_apply(&zaplist, &ft, o.baryv);
            if (nz < 0) {
                fprintf(stderr, "Error: failed to apply zaplist to '%s'\n", path);
                exit_status = 1;
                fftfile_close(&ft);
                if (use_pipeline) pipeline_done(&pipeline);
                continue;
            }
            if (o.verbose || o.nfiles == 1)
                fprintf(stderr, "Info: zapped %d interval(s) in %s\n", nz, path);
        }

        if (o.plot_candidates[0]) {
            CandidateList pc = read_candidates_file(o.plot_candidates, ft.T);
            if ((int)pc.n > o.ncands) pc.n = (size_t)o.ncands;
            char *stem;
            if (o.plotstem[0]) {
                stem = xstrdup(o.plotstem);
            } else {
                stem = (char *)malloc(strlen(o.plot_candidates) + 12);
                snprintf(stem, strlen(o.plot_candidates) + 12, "%s_profiles",
                         o.plot_candidates);
            }
            int pages = plot_candidates_svg(&ft, &pc,
                                            o.nharms_set ? &params : NULL,
                                            o.threshold, stem,
                                            o.plotcols, o.plotrows);
            fprintf(stderr, "Info: Wrote candidate profile plots (pages=%d stem=%s "
                            "n=%zu)\n", pages, stem, pc.n);
            free(stem);
            candlist_free(&pc);
            fftfile_close(&ft);
            continue;
        }
        if (use_gpu >= 0)
            fprintf(stderr, "Info: Searching (file=%s T=%.10g nharms=%d decimations=%d "
                            "backend=GPU device=%d)\n",
                    path, ft.T, params.nharms, params.ndecim, use_gpu);
        else
            fprintf(stderr, "Info: Searching (file=%s T=%.10g nharms=%d decimations=%d "
                            "threads=%d backend=CPU)\n",
                    path, ft.T, params.nharms, params.ndecim, omp_get_max_threads());

        SearchParams *pp = &params;
        MetricStats ms;
        int use_ms = o.metricstats || o.normalize;
        if (use_ms) metricstats_init(&ms, 0.0, 64.0, 3200, 16);

        CandidateList cands = search_file(&ft, pp, o.lofreq, o.hifreq, o.lobin,
                                          o.blocksize, o.threshold, o.remove,
                                          o.drtol, o.harm_remove, o.numharm,
                                          progress_mode, use_ms ? &ms : NULL,
                                          o.normalize, o.verbose, !o.nowisdom,
                                          o.wisdomfile, &cache, 1.0, use_gpu);

        candlist_sort_metric_desc(&cands);
        if ((int)cands.n > o.ncands) cands.n = (size_t)o.ncands;
        CandidateList with_ducy = measure_ducy(&ft, &cands, pp);
        candlist_free(&cands);

        int name_count = list_mode ? (o.nfiles > 1 ? o.nfiles : 2) : o.nfiles;
        char *outfile = output_path(path, o.outfilenm, o.outdir, name_count);
        if (write_candidates(&with_ducy, outfile ? outfile : "", o.threshold) != 0)
            exit_status = 1;
        if (o.plot && with_ducy.n > 0) {
            char *stem = plot_stem_for(o.plotstem, outfile, path);
            int pages = plot_candidates_svg(&ft, &with_ducy, pp, o.threshold,
                                            stem ? stem : "profiles",
                                            o.plotcols, o.plotrows);
            fprintf(stderr, "Info: Wrote candidate profile plots (pages=%d stem=%s)\n",
                    pages, stem ? stem : "profiles");
            free(stem);
        }
        candlist_free(&with_ducy);

        if (use_ms) {
            char *base;
            if (o.outfilenm[0]) {
                base = xstrdup(o.outfilenm);
            } else if (outfile && *outfile) {
                base = xstrdup(outfile);
                char *dot = strrchr(base, '.');
                if (dot && strcmp(dot, ".cohout") == 0) *dot = '\0';
            } else {
                base = base_stem(ft.path);
            }
            report_metricstats(&ms, base ? base : "candidates", pp, o.threshold);
            free(base);
            metricstats_free(&ms);
        }
        free(outfile);

        fftfile_close(&ft);
        if (use_pipeline) pipeline_done(&pipeline);
    } }

    if (use_pipeline) pipeline_free(&pipeline);
    search_cache_free(&cache);
    datpipe_free(pipe);
    if (have_zap) zaplist_free(&zaplist);
    if (o.files_owned) {
        { int i; for (i = 0; i < o.nfiles; i++) free(o.files[i]); }
    }
    free(o.outfilenm);
    free(o.outdir);
    free(o.wisdomfile);
    free(o.plotstem);
    free(o.plot_candidates);
    free(o.zaplist);
    free(o.files);
    return exit_status;
}

int cs_main(int argc, char **argv) { return cs_main_mode(argc, argv, 0); }
int cs_main_list(int argc, char **argv) { return cs_main_mode(argc, argv, 1); }
