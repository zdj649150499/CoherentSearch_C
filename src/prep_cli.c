/* Command-line glue for prep_coherent_search. */
#include "coherent_search.h"
#include "prep_search.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int prep_dedisp_main(int argc, char **argv);
extern void prepsubband_usage_print(FILE *f);

static int parse_int(const char *s, int *v)
{
    char *e = NULL;
    long x = strtol(s, &e, 10);
    if (!e || *e) return -1;
    *v = (int)x;
    return 0;
}

static int parse_double(const char *s, double *v)
{
    char *e = NULL;
    double x = strtod(s, &e);
    if (!e || *e) return -1;
    *v = x;
    return 0;
}

static const char *need_value(int argc, char **argv, int *i, const char *opt)
{
    if (*i + 1 >= argc) {
        fprintf(stderr, "Error: %s needs a value\n", opt);
        return NULL;
    }
    return argv[++(*i)];
}

static int parse_search_opts(int argc, char **argv, PrepSearchOpts *o,
                             char **files, int maxfiles, int *nfiles)
{
    memset(o, 0, sizeof(*o));
    *nfiles = 0;
    o->ncands = 100;
    o->lobin = 100;
    o->nharms = 60;
    o->m = 16;
    o->maxdecim = 6;
    o->numharm = 16;
    o->remove = 1;
    o->harm_remove = 1;
    o->threads = 0;
    o->gpu_device = -1;
    o->blocksize = 0;
    o->precision = CS_PREC_F32;
    o->sigma = CS_SIGMA_ANALYTIC;
    o->noprogress = 0;
    o->wisdom = 1;
    o->threshold = 8.0;
    o->lofreq = 0.1;
    o->hifreq = 125.0;
    o->hidr = 0.5;
    o->drtol = 1.0;
    o->baryv = 0.0;
    int i;
    for (i = 0; i < argc; i++) {
        const char *a = argv[i];
        const char *v;
        if (a[0] != '-' || a[1] == '\0') {
            if (*nfiles >= maxfiles) {
                fprintf(stderr, "Error: too many input files after ++\n");
                return -2;
            }
            files[(*nfiles)++] = argv[i];
            continue;
        }
        if (!strcmp(a, "-t") || !strcmp(a, "--threshold")) {
            if (!(v = need_value(argc, argv, &i, a)) ||
                parse_double(v, &o->threshold)) goto bad;
        } else if (!strcmp(a, "-n") || !strcmp(a, "--nharms")) {
            if (!(v = need_value(argc, argv, &i, a)) ||
                parse_int(v, &o->nharms)) goto bad;
        } else if (!strcmp(a, "--ncands")) {
            if (!(v = need_value(argc, argv, &i, a)) ||
                parse_int(v, &o->ncands)) goto bad;
        } else if (!strcmp(a, "--lobin")) {
            if (!(v = need_value(argc, argv, &i, a)) ||
                parse_int(v, &o->lobin)) goto bad;
        } else if (!strcmp(a, "--lofreq")) {
            if (!(v = need_value(argc, argv, &i, a)) ||
                parse_double(v, &o->lofreq)) goto bad;
        } else if (!strcmp(a, "--hifreq")) {
            if (!(v = need_value(argc, argv, &i, a)) ||
                parse_double(v, &o->hifreq)) goto bad;
        } else if (!strcmp(a, "--hidr")) {
            if (!(v = need_value(argc, argv, &i, a)) ||
                parse_double(v, &o->hidr)) goto bad;
        } else if (!strcmp(a, "--m")) {
            if (!(v = need_value(argc, argv, &i, a)) ||
                parse_int(v, &o->m)) goto bad;
        } else if (!strcmp(a, "--maxdecim")) {
            if (!(v = need_value(argc, argv, &i, a)) ||
                parse_int(v, &o->maxdecim)) goto bad;
        } else if (!strcmp(a, "--numharm")) {
            if (!(v = need_value(argc, argv, &i, a)) ||
                parse_int(v, &o->numharm)) goto bad;
        } else if (!strcmp(a, "--blocksize")) {
            if (!(v = need_value(argc, argv, &i, a)) ||
                parse_int(v, &o->blocksize)) goto bad;
        } else if (!strcmp(a, "--drtol")) {
            if (!(v = need_value(argc, argv, &i, a)) ||
                parse_double(v, &o->drtol)) goto bad;
        } else if (!strcmp(a, "--threads")) {
            if (!(v = need_value(argc, argv, &i, a)) ||
                parse_int(v, &o->threads)) goto bad;
        } else if (!strcmp(a, "--outdir")) {
            if (!(v = need_value(argc, argv, &i, a))) goto bad;
            snprintf(o->outdir, sizeof(o->outdir), "%s", v);
        } else if (!strcmp(a, "--wisdomfile")) {
            if (!(v = need_value(argc, argv, &i, a))) goto bad;
            snprintf(o->wisdomfile, sizeof(o->wisdomfile), "%s", v);
        } else if (!strcmp(a, "--zaplist")) {
            if (!(v = need_value(argc, argv, &i, a))) goto bad;
            snprintf(o->zaplist, sizeof(o->zaplist), "%s", v);
        } else if (!strcmp(a, "--baryv")) {
            if (!(v = need_value(argc, argv, &i, a)) ||
                parse_double(v, &o->baryv)) goto bad;
        } else if (!strcmp(a, "--precision")) {
            if (!(v = need_value(argc, argv, &i, a))) goto bad;
            if (!strcmp(v, "f32")) o->precision = CS_PREC_F32;
            else if (!strcmp(v, "f64")) o->precision = CS_PREC_F64;
            else goto bad;
        } else if (!strcmp(a, "--sigma")) {
            if (!(v = need_value(argc, argv, &i, a))) goto bad;
            if (!strcmp(v, "analytic")) o->sigma = CS_SIGMA_ANALYTIC;
            else if (!strcmp(v, "measured")) o->sigma = CS_SIGMA_MEASURED;
            else goto bad;
        } else if (!strcmp(a, "--gpu")) {
            if (i + 1 < argc && argv[i + 1][0] != '-' &&
                parse_int(argv[i + 1], &o->gpu_device) == 0) {
                i++;
            } else {
                o->gpu_device = 0;
            }
        } else if (!strcmp(a, "--noremove")) {
            o->remove = 0;
        } else if (!strcmp(a, "--noharmremove")) {
            o->harm_remove = 0;
        } else if (!strcmp(a, "--noprogress")) {
            o->noprogress = 1;
        } else if (!strcmp(a, "--nowisdom")) {
            o->wisdom = 0;
        } else if (!strcmp(a, "--progressbar")) {
            o->noprogress = 0;
        } else {
            fprintf(stderr, "Error: unknown search option '%s'\n", a);
            return -2;
        }
    }
    if (o->gpu_device >= 0) {
        char err[256] = {0};
        int g = cs_gpu_available(o->gpu_device, err, sizeof(err));
        if (g == 0) {
            fprintf(stderr, "Warning: %s; using CPU\n", err);
            o->gpu_device = -1;
        } else if (g < 0) {
            fprintf(stderr, "Error: %s\n", err);
            return -2;
        }
    }
    return 0;
bad:
    fprintf(stderr, "Error: bad value for search option '%s'\n", argv[i]);
    return -2;
}

/* Run one prepsubband-style command (argv[0] is a program name) plus the
 * search options after a '++' separator. */
int prep_run_command(int argc, char **argv)
{
    int sep = -1;
    int i;
    for (i = 1; i < argc; i++)
        if (!strcmp(argv[i], "++")) { sep = i; break; }
    int prep_end = sep < 0 ? argc : sep;
    int search_argc = sep < 0 ? 0 : argc - sep - 1;
    char **search_argv = sep < 0 ? NULL : argv + sep + 1;
    for (i = 0; i < search_argc; i++)
        if (!strcmp(search_argv[i], "-h") || !strcmp(search_argv[i], "--help")) {
            prep_print_help();
            return 0;
        }
    for (i = 1; i < prep_end; i++) {
        if (!strcmp(argv[i], "-sub")) {
            fprintf(stderr, "Error: -sub (subband output) cannot be searched; "
                            "use a time-series de-dispersion.\n");
            return 2;
        }
#ifndef CS_HAVE_CUDA
        if (!strcmp(argv[i], "-cuda")) {
            fprintf(stderr, "Error: this GPU=0 build has no CUDA support; "
                            "remove -cuda (CPU dedispersion is used).\n");
            return 2;
        }
#endif
    }

    PrepSearchOpts opts;
    char *extra_files[256];
    int n_extra = 0;
    int rc = parse_search_opts(search_argc, search_argv, &opts,
                               extra_files, 256, &n_extra);
    if (rc != 0) return 2;
    prep_search_set_opts(&opts);

    /* Files may be written after ++.  Move those positional arguments to the
     * prepsubband input list. */
    int cuda_seen = 0, cuda_dev = -1;
    for (i = 1; i < prep_end; i++) {
        if (!strcmp(argv[i], "-cuda")) {
            cuda_seen = 1;
            cuda_dev = -1;
            if (i + 1 < prep_end && parse_int(argv[i + 1], &cuda_dev) == 0)
                i++;
        }
    }
    int add_cuda = 0;
    char cuda_arg[32];
    if (opts.gpu_device >= 0) {
        if (cuda_seen) {
            if (cuda_dev >= 0 && cuda_dev != opts.gpu_device) {
                fprintf(stderr, "Error: -cuda %d conflicts with --gpu %d; "
                                "use the same CUDA device for both stages\n",
                        cuda_dev, opts.gpu_device);
                return 2;
            }
        } else {
            add_cuda = 1;
            snprintf(cuda_arg, sizeof(cuda_arg), "%d", opts.gpu_device);
            fprintf(stderr, "Info: --gpu %d also enables GPU de-dispersion "
                            "on the same device\n", opts.gpu_device);
        }
    }

    int new_argc = prep_end + n_extra + (add_cuda ? 2 : 0);
    char **new_argv = (char **)calloc((size_t)new_argc + 1, sizeof(char *));
    if (!new_argv) {
        fprintf(stderr, "Error: out of memory building the prepsubband command\n");
        return 2;
    }
    int k = 0;
    for (i = 0; i < prep_end; i++) new_argv[k++] = argv[i];
    if (add_cuda) {
        new_argv[k++] = "-cuda";
        new_argv[k++] = cuda_arg;
    }
    for (i = 0; i < n_extra; i++) new_argv[k++] = extra_files[i];
    new_argv[k] = NULL;

    rc = prep_dedisp_main(new_argc, new_argv);
    free(new_argv);
    return rc;
}

void prep_print_help(void)
{
    printf(
"prep_coherent_search -- PRESTO prepsubband de-dispersion followed by the\n"
"                       coherent harmonic search, all in memory (.dat is\n"
"                       never written).\n"
"\n"
"Usage:\n");
    printf("  prep_coherent_search [prepsubband options] -o OUTPREFIX [RAW...]\n"
           "                       ++ [search options] [RAW...]\n");
    printf(
"\n"
"Options before ++ are PRESTO prepsubband options.  Options after ++ are the\n"
"coherent search options of coherent_search.  The complete help for both is\n"
"printed below.\n"
"\n"
"================ prepsubband options (before ++) ================\n");
    prepsubband_usage_print(stdout);
    printf("\n\n"
"============ coherent search options (after ++) ============\n");
    cs_usage_mode(stdout, 0, 0);
    printf(
"\n"
"Prep notes:\n"
"  Raw input files may be placed before ++ or after ++; non-option names\n"
"  after ++ are appended to the prepsubband input list.\n"
"  With --gpu N after ++, de-dispersion and the search both use CUDA\n"
"  device N; the device is selected once and then reused.  If an\n"
"  explicit -cuda M is also present, M must equal N.\n"
"  -sub is rejected (subband output cannot be searched).\n"
"  -outlist is ignored; prep writes <OUTPREFIX>_DM<dm>.inf and\n"
"  <OUTPREFIX>_DM<dm>.cohout only, never a .dat or .dat.list.\n"
#ifndef CS_HAVE_CUDA
"  -cuda is unavailable in this GPU=0 build (CPU de-dispersion only);\n"
"  --gpu also falls back to the CPU.\n"
#else
"  --gpu N runs both de-dispersion and search on GPU N; an explicit\n"
"  -cuda N is accepted as an equivalent and must match N.\n"
#endif
"\n"
"Examples (RAW input may be placed before or after ++):\n"
"\n"
"  # CPU search of a PSRFITS observation\n"
"  prep_coherent_search -numout 851968 -nsub 224 -lodm 1002.5 \\\n"
"      -dmstep 0.5 -numdms 10 -downsamp 8 -o OUT OBS.fits \\\n"
"      ++ -t 6 -n 60 --maxdecim 16\n"
"\n"
"  # cached PSRFITS with mask, GPU de-dispersion + GPU search,\n"
"  # and the raw input file placed after ++\n"
"  prep_coherent_search -numout 6815744 -nsub 32 -lodm 5.0 \\\n"
"      -dmstep 0.05 -numdms 50 -downsamp 1 -zerodm -cache \\\n"
"      -mask OBS_rfifind.mask -o OUT \\\n"
"      ++ -t 6 -n 60 --maxdecim 16 --gpu 0 --blocksize 8192 \\\n"
"         --zaplist Lband.zaplist --lofreq 0.03 OBS.fits\n"
"\n"
"  # SIGPROC filterbank input, CPU search\n"
"  prep_coherent_search -numout 65536 -nsub 4 -lodm 10 -dmstep 1 \\\n"
"      -numdms 1 -downsamp 1 -nobary -o OUT test.fil \\\n"
"      ++ -t 5 -n 30 --maxdecim 4 --lofreq 0.1\n"
"\n"
"Output: OUT_DM<dm>.inf and OUT_DM<dm>.cohout; no .dat is written.\n");
}
