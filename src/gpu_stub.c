/* Stubs used when the binary is built without CUDA (make GPU=0). */
#include "coherent_search.h"

#include <stdio.h>
#include <string.h>

int cs_gpu_available(int device, char *errbuf, size_t errlen)
{
    (void)device;
    if (errbuf && errlen)
        snprintf(errbuf, errlen,
                 "this build has no CUDA support (rebuild with GPU=1)");
    return 0;
}

GpuSession *cs_gpu_session_create(const FFTFile *ft, const SearchParams *p,
                                  const DirectPlanSet *dplans,
                                  const long long *cstarts, long long total,
                                  int Nprof, long long nchunks,
                                  double r_lo, double lodr, int device)
{
    (void)ft; (void)p; (void)dplans; (void)cstarts; (void)total;
    (void)Nprof; (void)nchunks; (void)r_lo; (void)lodr; (void)device;
    return NULL;
}
int cs_gpu_session_matches(const GpuSession *s, const FFTFile *ft,
                           const SearchParams *p, int Nprof, long long total,
                           double r_lo, double lodr, int device)
{
    (void)s; (void)ft; (void)p; (void)Nprof; (void)total;
    (void)r_lo; (void)lodr; (void)device;
    return 0;
}
CandidateList cs_gpu_session_run(GpuSession *s, const FFTFile *ft,
                                 double threshold)
{
    (void)s; (void)ft; (void)threshold;
    CandidateList empty;
    candlist_init(&empty);
    return empty;
}
void cs_gpu_session_free(GpuSession *s) { (void)s; }

CandidateList cs_gpu_search_region(const FFTFile *ft, const SearchParams *p,
                                   const DirectPlanSet *dplans,
                                   const long long *cstarts, long long total,
                                   int Nprof, long long nchunks,
                                   double r_lo, double r_hi, double lodr,
                                   double threshold,
                                   int device)
{
    (void)ft; (void)p; (void)dplans; (void)cstarts; (void)total;
    (void)Nprof; (void)nchunks; (void)r_lo; (void)r_hi; (void)lodr;
    (void)threshold; (void)device;
    CandidateList empty;
    candlist_init(&empty);
    return empty;
}
