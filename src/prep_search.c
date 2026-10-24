/* Search hook for prep_coherent_search: dedispersed time series live in
 * memory; this file runs the ordinary CPU/GPU coherent search on each DM. */
#include "coherent_search.h"
#include "prep_search.h"

#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

static PrepSearchOpts g_opts;

void prep_search_set_opts(const PrepSearchOpts *opts)
{
    if (opts) g_opts = *opts;
}

static void ensure_dir(const char *path)
{
    if (!path || !*path) return;
    if (mkdir(path, 0777) != 0 && errno != EEXIST)
        fprintf(stderr, "Warning: cannot create output directory '%s': %s\n",
                path, strerror(errno));
}

static void candidate_path(char *out, size_t cap, const char *outfile,
                           int dmprecision, double dm)
{
    const char *base = outfile;
    if (g_opts.outdir[0]) {
        const char *slash = strrchr(outfile, '/');
        if (slash) base = slash + 1;
        snprintf(out, cap, "%s/%s_DM%.*f.cohout", g_opts.outdir, base,
                 dmprecision, dm);
    } else {
        snprintf(out, cap, "%s_DM%.*f.cohout", base, dmprecision, dm);
    }
}

int prep_search_run(const char *outfile, int dmprecision,
                    const double *dms, int ndms, long long N, double dt,
                    char **bufs, size_t *sizes)
{
    if (!outfile || !dms || !bufs || !sizes || ndms <= 0 || N <= 0 || dt <= 0.0)
        return -1;

    ensure_dir(g_opts.outdir);
    if (g_opts.threads > 0) omp_set_num_threads(g_opts.threads);

    SearchParams p;
    params_defaults(&p);
    p.nharms = g_opts.nharms > 0 ? g_opts.nharms : p.nharms;
    p.m = g_opts.m > 0 ? g_opts.m : p.m;
    p.hidr = g_opts.hidr > 0.0 ? g_opts.hidr : p.hidr;
    p.threshold = g_opts.threshold;
    p.precision = (cs_precision)g_opts.precision;
    p.sigma = (cs_sigma)g_opts.sigma;
    p.ndecim = decimation_set(p.nharms, g_opts.maxdecim, p.decimations, 64);

    int blocksize = g_opts.blocksize;
    if (blocksize <= 0 && g_opts.gpu_device >= 0) blocksize = 65536;

    ZapList zap;
    memset(&zap, 0, sizeof(zap));
    int have_zap = g_opts.zaplist[0] &&
                   zaplist_read(g_opts.zaplist, &zap) == 0;

    SearchCache cache;
    memset(&cache, 0, sizeof(cache));
    DatPipe *pipe = datpipe_create(6, 100, 6.0);
    if (!pipe) {
        fprintf(stderr, "Error: cannot create the realfft/rednoise pipeline\n");
        if (have_zap) zaplist_free(&zap);
        return -1;
    }

    int rc = 0;
    int i;
    for (i = 0; i < ndms; i++) {
        char infpath[8192];
        snprintf(infpath, sizeof(infpath), "%s_DM%.*f.inf",
                 outfile, dmprecision, dms[i]);
        if (!bufs[i] || sizes[i] < (size_t)N * sizeof(float)) {
            fprintf(stderr, "Error: short in-memory time series for DM %.*f\n",
                    dmprecision, dms[i]);
            rc = -1;
            continue;
        }
        FFTFile ft;
        if (datpipe_redfft_mem(pipe, (const float *)bufs[i], N, dt, infpath,
                               &ft) != 0) {
            fprintf(stderr, "Error: realfft/rednoise failed for DM %.*f\n",
                    dmprecision, dms[i]);
            rc = -1;
            continue;
        }
        {
            SimpleInf inf;
            if (inf_read(infpath, &inf) == 0) {
                inf_free(&ft.inf);
                ft.inf = inf;
            } else {
                ft.inf.has_DM = 1;
                ft.inf.DM = dms[i];
            }
        }
        if (have_zap) {
            int nz = zaplist_apply(&zap, &ft, g_opts.baryv);
            if (nz < 0) {
                fprintf(stderr, "Error: zaplist failed for DM %.*f\n",
                        dmprecision, dms[i]);
                fftfile_close(&ft);
                rc = -1;
                continue;
            }
        }

        CandidateList cands = search_file(
            &ft, &p, g_opts.lofreq, g_opts.hifreq, g_opts.lobin, blocksize,
            g_opts.threshold, g_opts.remove, g_opts.drtol, g_opts.harm_remove,
            g_opts.numharm, NULL, 0, 0, g_opts.wisdom,
            g_opts.wisdomfile[0] ? g_opts.wisdomfile : NULL, &cache, 1.0,
            g_opts.gpu_device);
        candlist_sort_metric_desc(&cands);
        if (g_opts.ncands > 0 && cands.n > (size_t)g_opts.ncands)
            cands.n = (size_t)g_opts.ncands;
        CandidateList with_ducy = measure_ducy(&ft, &cands, &p);
        candlist_free(&cands);

        char outpath[8192];
        candidate_path(outpath, sizeof(outpath), outfile, dmprecision, dms[i]);
        if (write_candidates(&with_ducy, outpath, g_opts.threshold) != 0)
            rc = -1;
        candlist_free(&with_ducy);
        fftfile_close(&ft);
    }
    search_cache_free(&cache);
    datpipe_free(pipe);
    if (have_zap) zaplist_free(&zap);
    return rc;
}
