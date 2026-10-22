#ifndef PRESTO_GPU_CPU_H
#define PRESTO_GPU_CPU_H
/* CPU-only compatibility layer for the vendored prepsubband sources.
 * The GPU branches are compiled but never called when cmd->cudaP == 0;
 * these stubs keep the code linkable without any CUDA toolkit/device. */
#include <stddef.h>

typedef void *cudaStream_t;
typedef int cudaError_t;
enum { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2,
       cudaMemcpyDeviceToDevice = 3 };
enum { cudaHostAllocMapped = 1, cudaHostAllocWriteCombined = 2 };

int  cudaMalloc(void **p, size_t n);
int  cudaFree(void *p);
int  cudaMallocHost(void **p, size_t n);
int  cudaFreeHost(void *p);
int  cudaMemcpy(void *dst, const void *src, size_t n, int kind);
int  cudaMemset(void *p, int v, size_t n);
int  cudaHostAlloc(void **p, size_t n, unsigned flags);
int  cudaHostGetDevicePointer(void **dptr, void *hptr, unsigned flags);
int  cudaStreamSynchronize(cudaStream_t stream);

void select_cuda_dev(int dev);
void Endup_GPU(void);
void free_prep_subband_GPU_cache_array(void);
float get_mean_gpu(float *indata, int n);
double get_mean_gpu_d(double *indata, int n);
float get_variance_gpu(float *indata, float mean, int n);
float gpu_sum_reduce_float(float *arr, int n);
float gpu_varience_reduce_float(float *arr, float mean, int n);
void dedisp_subbands_GPU(float *lastdata_gpu, float *lastdata, int numpts,
                         int numchan, int *delays, int numsubbands,
                         float *result, int transpose);
void dedisp_subbands_GPU_cache(unsigned char *data, float *scl, float *offs,
                               unsigned char *last, float *last_scl,
                               float *last_offs, int numpts, int numchan,
                               int *delays, int numsubbands, float *result,
                               int transpose);
void dedisp_subbands_GPU_cache_stream(unsigned char *data, float *scl,
                                      float *offs, unsigned char *last,
                                      float *last_scl, float *last_offs,
                                      int numpts, int numchan, int *delays,
                                      int numsubbands, float *result,
                                      int transpose, cudaStream_t stream);
void downsamp_GPU(float *in, float *out, int numchan, int numpts, int down,
                  int transpose);
void float_dedisp_GPU(float *data, float *last, int numpts, int numchan,
                      int *delays, float approx_mean, float *result,
                      int numdms, int transpose);
void Get_subsdata(float *in, short *out, int nsub, int worklen);
void SetSubData4Mask_GPU(float *subbanddata, float *padvals, int *maskchans,
                         int len, int width);
void ZeroDM_subchan_GPU(float *indata, int len, int width);
void transpose_GPU(float *src, float *dst, int width, int height);
void transpose_GPU_short(short *src, float *dst, int width, int height);

#endif
