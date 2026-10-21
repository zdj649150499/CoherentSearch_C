/* PRESTO .inf / .fft readers. */
#include "coherent_search.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static char *cs_strdup(const char *s)
{
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *r = (char *)malloc(n);
    if (r) memcpy(r, s, n);
    return r;
}

static char *trim(char *s)
{
    while (*s && isspace((unsigned char)*s)) s++;
    char *e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) *--e = '\0';
    return s;
}

/* Julia: strip(split(line, "=")[end]) */
static char *inf_value(char *line)
{
    char *eq = strrchr(line, '=');
    if (!eq) return trim(line);
    return trim(eq + 1);
}

int inf_read(const char *path, SimpleInf *inf)
{
    memset(inf, 0, sizeof(*inf));
    inf->path = cs_strdup(path);

    FILE *f = fopen(path, "r");
    if (!f) return -1;

    char *line = NULL;
    size_t cap = 0;
    ssize_t len;
    while ((len = getline(&line, &cap, f)) >= 0) {
        (void)len;
        /* Julia startswith() on the raw line, including its leading space. */
        if (strncmp(line, " Object being observed", 22) == 0) {
            free(inf->object);
            inf->object = cs_strdup(inf_value(line));
        } else if (strncmp(line, " Epoch", 6) == 0) {
            inf->epoch = strtod(inf_value(line), NULL);
            inf->has_epoch = 1;
        } else if (strncmp(line, " Number of bins", 15) == 0) {
            inf->N = strtoll(inf_value(line), NULL, 10);
            inf->has_N = 1;
        } else if (strncmp(line, " Width of each time series bin", 30) == 0) {
            inf->dt = strtod(inf_value(line), NULL);
            inf->has_dt = 1;
        } else if (strncmp(line, " Dispersion measure", 19) == 0) {
            inf->DM = strtod(inf_value(line), NULL);
            inf->has_DM = 1;
        }
    }
    free(line);
    fclose(f);
    return 0;
}

void inf_free(SimpleInf *inf)
{
    if (!inf) return;
    free(inf->path);
    free(inf->object);
    memset(inf, 0, sizeof(*inf));
}

int fftfile_open(FFTFile *ft, const char *path)
{
    memset(ft, 0, sizeof(*ft));
    ft->path = cs_strdup(path);

    /* string(chop(path; tail=4), ".inf") */
    size_t plen = strlen(path);
    size_t keep = plen > 4 ? plen - 4 : 0;
    char *infpath = (char *)malloc(keep + 5);
    if (!infpath) return -1;
    memcpy(infpath, path, keep);
    memcpy(infpath + keep, ".inf", 5);

    if (inf_read(infpath, &ft->inf) != 0) {
        fprintf(stderr, "Error: The .inf file '%s' was not found.\n", infpath);
        free(infpath);
        return -1;
    }
    free(infpath);

    if (!ft->inf.has_N) {
        fprintf(stderr, "Error: Missing 'Number of bins' in %s\n", ft->inf.path);
        return -1;
    }
    if (!ft->inf.has_dt) {
        fprintf(stderr, "Error: Missing 'Width of each time series bin' in %s\n",
                ft->inf.path);
        return -1;
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Error: cannot open '%s': %s\n", path, strerror(errno));
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "Error: cannot stat '%s': %s\n", path, strerror(errno));
        close(fd);
        return -1;
    }
    size_t bytes = (size_t)st.st_size;
    ft->mapsize = bytes;
    if (bytes == 0) {
        ft->amps = NULL;
        ft->namps = 0;
    } else {
        /* MAP_PRIVATE so a zaplist can modify the in-memory spectrum without
         * ever touching the input file.  Unmodified pages stay shared. */
        void *m = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) {
            fprintf(stderr, "Error: cannot mmap '%s': %s\n", path, strerror(errno));
            close(fd);
            return -1;
        }
        ft->map = m;
        ft->amps = (float *)m;
        ft->namps = (long long)(bytes / sizeof(float) / 2);
        ft->owns_amps = 0;
    }
    close(fd);

    ft->N = ft->inf.N;
    ft->T = (double)ft->N * ft->inf.dt;
    ft->df = 1.0 / ft->T;
    ft->dereddened = strstr(path, "_red.fft") != NULL;
    ft->detrended = ft->dereddened;
    if (ft->namps > 0) {
        ft->DC = ft->amps[0];
        ft->Nyquist = ft->amps[1];
    }
    return 0;
}

void fftfile_close(FFTFile *ft)
{
    if (!ft) return;
    if (ft->map && ft->mapsize) munmap(ft->map, ft->mapsize);
    else if (ft->owns_amps) free(ft->amps);
    inf_free(&ft->inf);
    free(ft->path);
    memset(ft, 0, sizeof(*ft));
}
