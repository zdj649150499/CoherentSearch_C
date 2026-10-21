/*
 * Internal declarations for the CUDA backend.  Included only by cuda.cu.
 */
#ifndef COHERENT_SEARCH_CUDA_CUH
#define COHERENT_SEARCH_CUDA_CUH

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdint.h>

/* Device mirror of one DirectPlan (pointers refer to device copies). */
struct GpuPlan {
    int h, m, q, s, base_adv, pnum, rfloor0, P;
    int32_t *row;
    float *W;
    float *A_re, *A_im;
    uint8_t *carry;
    int ngrp;
    int32_t *grow, *goff, *gnj;
    float *gW, *gA_re, *gA_im;
};

#endif
