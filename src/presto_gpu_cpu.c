/* Link-time stubs for prepsubband's GPU branches when built without CUDA. */
#include <stddef.h>
#include "presto_gpu_cpu.h"

int cudaMalloc(void **p, size_t n) { (void)n; if (p) *p = NULL; return -1; }
int cudaFree(void *p) { (void)p; return 0; }
int cudaMallocHost(void **p, size_t n) { (void)n; if (p) *p = NULL; return -1; }
int cudaFreeHost(void *p) { (void)p; return 0; }
int cudaMemcpy(void *d, const void *s, size_t n, int k) { (void)d; (void)s; (void)n; (void)k; return -1; }
int cudaMemset(void *p, int v, size_t n) { (void)p; (void)v; (void)n; return -1; }
int cudaHostAlloc(void **p, size_t n, unsigned f) { (void)n; (void)f; if (p) *p = NULL; return -1; }
int cudaHostGetDevicePointer(void **d, void *h, unsigned f) { (void)h; (void)f; if (d) *d = NULL; return -1; }
int cudaStreamSynchronize(cudaStream_t s) { (void)s; return -1; }

void select_cuda_dev(int d) { (void)d; }
void Endup_GPU(void) {}
float get_mean_gpu(float *a, int n) { (void)a; (void)n; return 0.0f; }
double get_mean_gpu_d(double *a, int n) { (void)a; (void)n; return 0.0; }
float get_variance_gpu(float *a, float m, int n) { (void)a; (void)m; (void)n; return 0.0f; }
float gpu_sum_reduce_float(float *a, int n) { (void)a; (void)n; return 0.0f; }
float gpu_varience_reduce_float(float *a, float m, int n) { (void)a; (void)m; (void)n; return 0.0f; }
void dedisp_subbands_GPU(float *a, float *b, int c, int d, int *e, int f,
                         float *g, int h)
{ (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h; }
void dedisp_subbands_GPU_cache(unsigned char *a, float *b, float *c,
                               unsigned char *d, float *e, float *f, int g,
                               int h, int *i, int j, float *k, int l)
{ (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;(void)j;(void)k;(void)l; }
void dedisp_subbands_GPU_cache_stream(unsigned char *a, float *b, float *c,
                                      unsigned char *d, float *e, float *f,
                                      int g, int h, int *i, int j, float *k,
                                      int l, cudaStream_t s)
{ (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;(void)j;(void)k;(void)l;(void)s; }
void downsamp_GPU(float *a, float *b, int c, int d, int e, int f)
{ (void)a;(void)b;(void)c;(void)d;(void)e;(void)f; }
void float_dedisp_GPU(float *a, float *b, int c, int d, int *e, float f,
                      float *g, int h, int i)
{ (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i; }
void Get_subsdata(float *a, short *b, int c, int d) { (void)a;(void)b;(void)c;(void)d; }
void SetSubData4Mask_GPU(float *a, float *b, int *c, int d, int e)
{ (void)a;(void)b;(void)c;(void)d;(void)e; }
void ZeroDM_subchan_GPU(float *a, int b, int c) { (void)a;(void)b;(void)c; }
void transpose_GPU(float *a, float *b, int c, int d) { (void)a;(void)b;(void)c;(void)d; }
void transpose_GPU_short(short *a, float *b, int c, int d) { (void)a;(void)b;(void)c;(void)d; }
