/*
 * Zaplist support (PRESTO accelsearch -zaplist semantics).
 *
 * Each line of a zaplist holds a frequency (Hz) and width (Hz).  Lines
 * beginning with '#' are comments; lines beginning with 'B' are barycentric
 * frequencies and are used as-is.  Other lines are corrected by (1 + baryv).
 *
 * The bins are zapped exactly as PRESTO's zapbirds() does for an in-memory FFT:
 *   - lobin/hibin = (freq -/+ width/2) * T * scale, at least one bin wide;
 *   - sorted by lobin ascending;
 *   - bounds clamped to [1, N/2];
 *   - the mean amplitude is sqrt(0.5*(median_lo+median_hi)/-log(0.5)), where
 *     each median is over 200 bins of POWER = re^2+im^2 before/after the bird;
 *   - bins floor(lobin) .. ceil(hibin)-1 are set to (meanamp, 0).
 * Nothing is written to the input file: .fft amplitudes are mapped MAP_PRIVATE.
 */
#include "coherent_search.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ZAP_MEDIANBINS 200

typedef struct { double lobin, hibin; } ZapBin;

static int zapbin_cmp(const void *a, const void *b)
{
    const ZapBin *x = (const ZapBin *)a;
    const ZapBin *y = (const ZapBin *)b;
    return (x->lobin > y->lobin) - (x->lobin < y->lobin);
}

/* PRESTO median.c: Numerical-Recipes quickselect, lower median for even n.
 * Destroys the array. */
static float presto_median_lower(float arr[], int n)
{
    int low = 0, high = n - 1;
    int median = (low + high) / 2;
    for (;;) {
        if (high <= low) return arr[median];
        if (high == low + 1) {
            if (arr[low] > arr[high]) {
                float t = arr[low]; arr[low] = arr[high]; arr[high] = t;
            }
            return arr[median];
        }
        int middle = (low + high) / 2;
        if (arr[middle] > arr[high]) {
            float t = arr[middle]; arr[middle] = arr[high]; arr[high] = t;
        }
        if (arr[low] > arr[high]) {
            float t = arr[low]; arr[low] = arr[high]; arr[high] = t;
        }
        if (arr[middle] > arr[low]) {
            float t = arr[middle]; arr[middle] = arr[low]; arr[low] = t;
        }
        {
            float t = arr[middle]; arr[middle] = arr[low + 1]; arr[low + 1] = t;
        }
        int ll = low + 1, hh = high;
        for (;;) {
            do ll++; while (arr[low] > arr[ll]);
            do hh--; while (arr[hh] > arr[low]);
            if (hh < ll) break;
            float t = arr[ll]; arr[ll] = arr[hh]; arr[hh] = t;
        }
        {
            float t = arr[low]; arr[low] = arr[hh]; arr[hh] = t;
        }
        if (hh <= median) low = ll;
        if (hh >= median) high = hh - 1;
    }
}

/* Median power over `count` bins starting at `start`, clamped to the array. */
static float median_power(const float *amps, long long namps,
                          long long start, int count)
{
    if (start < 0) { count += (int)start; start = 0; }
    if (start + count > namps) count = (int)(namps - start);
    if (count <= 0) return 0.0f;
    float buf[ZAP_MEDIANBINS];
    { int i; for (i = 0; i < count; i++) {
        float r = amps[2 * (start + i)];
        float im = amps[2 * (start + i) + 1];
        buf[i] = r * r + im * im;
    } }
    return presto_median_lower(buf, count);
}

/* Zap one interval, matching PRESTO's in-memory zapbirds().  Returns 1 if it
 * changed bins, 0 if the interval was skipped. */
static int zap_one(float *amps, long long namps, double lobin, double hibin)
{
    long long ilobin = (long long)floor(lobin);
    long long ihibin = (long long)ceil(hibin);
    if (ilobin < 1) ilobin = 1;
    if (ihibin > namps) ihibin = namps;
    long long binstozap = ihibin - ilobin;
    if (binstozap <= 0) return 0;

    /* PRESTO only zaps birds this far from DC, because it needs 200-bin
     * median windows on both sides. */
    if (!(lobin - 1.5 * ZAP_MEDIANBINS > 1.0)) return 0;

    long long lo_start = (long long)(lobin - 3 * ZAP_MEDIANBINS / 2);
    long long hi_start = (long long)(hibin - ZAP_MEDIANBINS / 2);
    float median_lo = median_power(amps, namps, lo_start, ZAP_MEDIANBINS);
    float median_hi = median_power(amps, namps, hi_start, ZAP_MEDIANBINS);
    float avgamp = (float)sqrt(0.5 * (median_lo + median_hi) / -log(0.5));

    { long long i; for (i = ilobin; i < ihibin; i++) {
        amps[2 * i] = avgamp;
        amps[2 * i + 1] = 0.0f;
    } }
    return 1;
}

int zaplist_read(const char *path, ZapList *z)
{
    memset(z, 0, sizeof(*z));
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[1024];
    int cap = 0;
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        if (*p == '\n' || *p == '\r' || *p == '\0') continue;
        if (*p == '#') continue;
        int bary = 0;
        if (*p == 'B') { bary = 1; p++; }
        double freq = 0.0, width = 0.0;
        if (sscanf(p, "%lf %lf", &freq, &width) != 2) continue;
        if (!(freq > 0.0) || !(width > 0.0)) continue;
        if (z->n == cap) {
            cap = cap ? cap * 2 : 32;
            z->v = (ZapBird *)realloc(z->v, (size_t)cap * sizeof(ZapBird));
        }
        z->v[z->n].freq = freq;
        z->v[z->n].width = width;
        z->v[z->n].bary = bary;
        z->n++;
    }
    fclose(f);
    return 0;
}

void zaplist_free(ZapList *z)
{
    free(z->v);
    memset(z, 0, sizeof(*z));
}

int zaplist_apply(const ZapList *z, FFTFile *ft, double baryv)
{
    if (!z || z->n <= 0) return 0;
    ZapBin *bins = (ZapBin *)malloc((size_t)z->n * sizeof(ZapBin));
    if (!bins) return -1;
    { int i; for (i = 0; i < z->n; i++) {
        double scale = z->v[i].bary ? 1.0 : (1.0 + baryv);
        double f = z->v[i].freq;
        double w = z->v[i].width;
        double lo = (f - 0.5 * w) * ft->T * scale;
        double hi = (f + 0.5 * w) * ft->T * scale;
        if (hi - lo < 1.0) {
            double mid = 0.5 * (hi + lo);
            lo = mid - 0.5;
            hi = mid + 0.5;
        }
        bins[i].lobin = lo;
        bins[i].hibin = hi;
    } }
    qsort(bins, (size_t)z->n, sizeof(ZapBin), zapbin_cmp);

    long long hibin_limit = ft->N / 2;
    if (hibin_limit > ft->namps) hibin_limit = ft->namps;
    int applied = 0;
    { int i; for (i = 0; i < z->n; i++) {
        if (bins[i].lobin >= (double)hibin_limit) break;
        if (bins[i].hibin >= (double)hibin_limit)
            bins[i].hibin = (double)(hibin_limit - 1);
        applied += zap_one(ft->amps, ft->namps, bins[i].lobin, bins[i].hibin);
    } }
    free(bins);
    return applied;
}
