/* Candidate pulse-profile plots, dependency-free SVG output.
 *
 * Mirrors the layout of bin/plotting.jl: US-Letter portrait pages, a grid of
 * ncols x nrows panels, each profile rotated so its peak sits at phase 0.5,
 * and the same two-line captions plus page header.  The Julia version writes
 * PNG through CairoMakie; here the pages are self-contained SVG, which needs no
 * graphics library and can be converted (`rsvg-convert`, a browser, Inkscape).
 */
#include "coherent_search.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define PAGE_W 850
#define PAGE_H 1100
#define MARGIN_L 40
#define MARGIN_R 20
#define MARGIN_T 120
#define MARGIN_B 30

static void xml_escape(FILE *f, const char *s)
{
    if (!s) return;
    for (; *s; s++) {
        switch (*s) {
        case '&': fputs("&amp;", f); break;
        case '<': fputs("&lt;", f); break;
        case '>': fputs("&gt;", f); break;
        case '"': fputs("&quot;", f); break;
        default: fputc(*s, f);
        }
    }
}

static void mkdir_parent(const char *path)
{
    const char *slash = strrchr(path, '/');
    if (!slash || slash == path) return;
    size_t n = (size_t)(slash - path);
    char *dir = (char *)malloc(n + 1);
    if (!dir) return;
    memcpy(dir, path, n);
    dir[n] = '\0';
    { char *q; for (q = dir + 1; *q; q++) {
        if (*q == '/') {
            *q = '\0';
            mkdir(dir, 0777);
            *q = '/';
        }
    } }
    mkdir(dir, 0777);
    free(dir);
}

/* Julia circshift(prof, shift): result[i] = prof[i - shift]. */
static void rotate_to_peak(const double *prof, int n, double *out)
{
    int amax = 0;
    { int i; for (i = 1; i < n; i++) if (prof[i] > prof[amax]) amax = i; }
    int shift = n / 2 - amax;
    { int i; for (i = 0; i < n; i++) {
        int j = i - shift;
        j %= n;
        if (j < 0) j += n;
        out[i] = prof[j];
    } }
}

static const char *base_name_c(const char *path)
{
    const char *s = strrchr(path, '/');
    return s ? s + 1 : path;
}

int plot_candidates_svg(const FFTFile *ft, const CandidateList *cands,
                        const SearchParams *p, double threshold,
                        const char *outstem, int ncols, int nrows)
{
    if (!cands || cands->n == 0) return 0;
    if (ncols < 1) ncols = 3;
    if (nrows < 1) nrows = 5;
    const int per_page = ncols * nrows;
    const int npages = (int)((cands->n + per_page - 1) / per_page);
    mkdir_parent(outstem);

    char datebuf[64];
    time_t now = time(NULL);
    struct tm tmv;
    localtime_r(&now, &tmv);
    strftime(datebuf, sizeof(datebuf), "%Y-%m-%dT%H:%M:%S", &tmv);

    const char *obj = ft->inf.object ? ft->inf.object : "?";
    const int m = 64;

    { int page; for (page = 0; page < npages; page++) {
        long lo = (long)page * per_page;
        long hi = lo + per_page;
        if (hi > (long)cands->n) hi = (long)cands->n;
        int nfilled = (int)(hi - lo);
        int last_row = (nfilled + ncols - 1) / ncols;

        char fname[4096];
        snprintf(fname, sizeof(fname), "%s_%02d.svg", outstem, page + 1);
        FILE *f = fopen(fname, "w");
        if (!f) {
            fprintf(stderr, "Error: cannot write plot '%s'\n", fname);
            continue;
        }
        fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        fprintf(f, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" "
                   "height=\"%d\" viewBox=\"0 0 %d %d\" "
                   "font-family=\"Helvetica,Arial,sans-serif\">\n",
                2 * PAGE_W, 2 * PAGE_H, PAGE_W, PAGE_H);
        fprintf(f, "<rect width=\"100%%\" height=\"100%%\" fill=\"white\"/>\n");

        /* Page header */
        fprintf(f, "<text x=\"%d\" y=\"24\" font-size=\"11\" font-weight=\"bold\">",
                MARGIN_L);
        xml_escape(f, base_name_c(ft->path));
        fprintf(f, "      Candidate pulse profiles</text>\n");
        fprintf(f, "<text x=\"%d\" y=\"42\" font-size=\"11\">", MARGIN_L);
        fprintf(f, "object=");
        xml_escape(f, obj);
        if (ft->inf.has_DM)
            fprintf(f, "   DM=%.2f", ft->inf.DM);
        else
            fprintf(f, "   DM=?");
        fprintf(f, "   T=%.1f s   N=%lld", ft->T, ft->N);
        if (p) fprintf(f, "   nharms=%d   metric=boxcar", p->nharms);
        fprintf(f, "   threshold=%.1f", threshold);
        fprintf(f, "</text>\n");
        fprintf(f, "<text x=\"%d\" y=\"60\" font-size=\"11\">%s        Page %d of %d</text>\n",
                MARGIN_L, datebuf, page + 1, npages);

        const double gw = PAGE_W - MARGIN_L - MARGIN_R;
        const double gh = PAGE_H - MARGIN_T - MARGIN_B;
        const double cw = gw / ncols;
        const double ch = gh / nrows;

        { int slot; for (slot = 0; slot < per_page; slot++) {
            long idx = lo + slot;
            if (idx >= hi) break;
            const Candidate *c = &cands->v[idx];
            int row = slot / ncols;
            int col = slot % ncols;
            double x0 = MARGIN_L + col * cw;
            double y0 = MARGIN_T + row * ch;

            /* Panel plot area */
            const double pl = 34, pr = 8, pt = 30, pb = 22;
            double px = x0 + pl, py = y0 + pt;
            double pw = cw - pl - pr, ph = ch - pt - pb;

            /* Profile, folded at the full requested depth like Julia. */
            int nfold = p ? p->nharms : c->nharm;
            if (nfold < 1) nfold = c->nharm;
            int n = 0;
            double *prof = candidate_profile(ft, c->r, nfold, m, &n);
            if (!prof || n < 2) {
                free(prof);
                continue;
            }
            double *rot = (double *)malloc((size_t)n * sizeof(double));
            if (!rot) { free(prof); continue; }
            rotate_to_peak(prof, n, rot);
            free(prof);
            double ymin = rot[0], ymax = rot[0];
            { int i; for (i = 1; i < n; i++) {
                if (rot[i] < ymin) ymin = rot[i];
                if (rot[i] > ymax) ymax = rot[i];
            } }
            if (ymax <= ymin) { ymax = ymin + 1.0; ymin -= 1.0; }
            else { double pad = 0.08 * (ymax - ymin); ymin -= pad; ymax += pad; }

            /* Axes */
            fprintf(f, "<rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" "
                       "fill=\"none\" stroke=\"#999\" stroke-width=\"0.8\"/>\n",
                    px, py, pw, ph);
            fprintf(f, "<text x=\"%.1f\" y=\"%.1f\" font-size=\"9\">",
                    px, y0 + 12);
            fprintf(f, "#%ld    S/N %.2f    #Harm %d", idx + 1, c->metric, c->nharm);
            if (p && c->nharm > 0) fprintf(f, " (k=%d)", p->nharms / c->nharm);
            fprintf(f, "</text>\n");
            fprintf(f, "<text x=\"%.1f\" y=\"%.1f\" font-size=\"9\">",
                    px, y0 + 24);
            fprintf(f, "f = %.10f Hz    P = %.6f ms", c->freq, 1000.0 / c->freq);
            fprintf(f, "</text>\n");

            /* Profile line */
            fprintf(f, "<polyline fill=\"none\" stroke=\"#000080\" stroke-width=\"1.2\" points=\"");
            { int i; for (i = 0; i < n; i++) {
                double xv = (n == 1) ? 0.0 : (double)i / (double)n;
                double yv = (rot[i] - ymin) / (ymax - ymin);
                fprintf(f, "%.2f,%.2f ", px + xv * pw, py + (1.0 - yv) * ph);
            } }
            fprintf(f, "\"/>\n");

            /* x ticks and label */
            { int t; for (t = 0; t <= 2; t++) {
                double xv = 0.5 * t;
                fprintf(f, "<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" "
                           "stroke=\"#999\" stroke-width=\"0.6\"/>\n",
                        px + xv * pw, py + ph, px + xv * pw, py + ph + 4);
                fprintf(f, "<text x=\"%.1f\" y=\"%.1f\" font-size=\"8\" "
                           "text-anchor=\"middle\">%.1f</text>\n",
                        px + xv * pw, py + ph + 14, xv);
            } }
            if (row == last_row - 1)
                fprintf(f, "<text x=\"%.1f\" y=\"%.1f\" font-size=\"9\" "
                           "text-anchor=\"middle\">Pulse phase</text>\n",
                        px + 0.5 * pw, y0 + ch - 2);
            free(rot);
        } }
        fprintf(f, "</svg>\n");
        fclose(f);
    } }
    return npages;
}
