#ifndef COHERENT_SEARCH_PREP_SEARCH_H
#define COHERENT_SEARCH_PREP_SEARCH_H

#include <stddef.h>

/* Search options used after dedispersion.  precision/sigma use the
 * CS_PREC_* / CS_SIGMA_* values from coherent_search.h. */
typedef struct {
    int ncands;
    int lobin;
    int nharms;
    int m;
    int maxdecim;
    int numharm;
    int remove;
    int harm_remove;
    int threads;
    int gpu_device;       /* -1 = CPU */
    int blocksize;        /* 0 = backend default */
    int precision;
    int sigma;
    int noprogress;
    int wisdom;
    double threshold;
    double lofreq;
    double hifreq;
    double hidr;
    double drtol;
    double baryv;
    char outdir[4096];
    char wisdomfile[4096];
    char zaplist[4096];
} PrepSearchOpts;

void prep_search_set_opts(const PrepSearchOpts *opts);

/* Print the combined prepsubband + ++ search help. */
void prep_print_help(void);

/* Called by the dedispersion core after all DM time series are complete.
 * bufs[i] holds N floats for dms[i]; inf files already exist as
 * <outfile>_DM<precision><dm>.inf. */
int prep_search_run(const char *outfile, int dmprecision,
                    const double *dms, int ndms, long long N, double dt,
                    char **bufs, size_t *sizes);

#endif
