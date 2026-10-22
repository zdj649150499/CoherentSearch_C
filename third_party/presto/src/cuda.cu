#include "cuda.cuh"
#include <math_constants.h>


// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/partition.h>


#define MAX_BLOCK_SZ 512
#define BLOCK_DIM 16
#define WARP_SIZE 32
#define MAX_THREAD 1024
#define NEAREST_INT(x) (int) (x<0 ? x-0.5 : x+0.5)
#define ACCEL_RDZ 0.5
#define DBLCORRECT    1e-14

/* Simple linear interpolation macro */
#define LININTERP(X, xlo, xhi, ylo, yhi) \
    ((ylo)+((X)-(xlo))*((yhi)-(ylo))/((xhi)-(xlo)))

__constant__  gpu_parameter_add c_gpu_parameter_add[64];
__constant__ float c_norm_data[64];  // 假设最大64个文件

// 可调参数：每个 block 的线程数（TILE_X）和 y tile 大小（YTILE）
// TILE_X 应为 32 的倍数以便更好利用 warp；一般 128/256/512 常见
// YTILE 取 4/8/16，注意 shared memory 大小 = TILE_X * YTILE * sizeof(cufftComplex)
#ifndef TILE_X
  #define TILE_X 512
#endif
#ifndef YTILE
  #define YTILE 8
#endif

#define ACCEL_PADDING 2000

typedef float2 Complex;

//define a texture memory
// texture<cufftComplex> tex_d_kernel;
// texture<cufftComplex> tex_d_fpdata;
// texture<cufftComplex> tex_d_fdata;
// texture<unsigned short> tex_d_zinds;
// texture<unsigned short> tex_d_rinds;
// texture<float> tex_d_fundamental;

typedef struct kernel{
    int z;               /* The fourier f-dot of the kernel */
    int w;               /* The fourier f-dot-dot of the kernel */
    int fftlen;          /* Number of complex points in the kernel */
    int numgoodbins;     /* The number of good points you can get back */
    int numbetween;      /* Fourier freq resolution (2=interbin) */
    int kern_half_width; /* Half width (bins) of the raw kernel. */
    fcomplex *data;      /* The FFTd kernel itself */
} kernel;

typedef struct subharminfo{
    int numharm;       /* The number of sub-harmonics */
    int harmnum;       /* The sub-harmonic number (fundamental = numharm) */
    int zmax;          /* The maximum Fourier f-dot for this harmonic */
    int wmax;          /* The maximum Fourier f-dot-dot for this harmonic */
    int numkern_zdim;  /* Number of kernels calculated in the z dimension */
    int numkern_wdim;  /* Number of kernels calculated in the w dimension */
    int numkern;       /* Total number of kernels in the vector == numzs */
    kernel **kern;     /* A 2D array of the kernels themselves, with dimensions of z and w */
    unsigned short *rinds; /* Table of lookup indices for Fourier Freqs: subharmonic r values corresponding to "fundamental" r values */
    unsigned short *zinds; /* Table of lookup indices for Fourier F-dots */
} subharminfo;

void cp_gpu_parameter_add_list_to_const(gpu_parameter_add *gpu_parameter_add_list_cpu, int num)
{
    cudaMemcpyToSymbol(c_gpu_parameter_add, gpu_parameter_add_list_cpu, sizeof(gpu_parameter_add) * num, 0, cudaMemcpyHostToDevice);
}


void cp_norm_data_to_const(float *norm_data, int num)
{
    /* Asynchronous (stream-ordered) copy: on the default stream each
     * copy is serialized w.r.t. the surrounding kernel launches, so every
     * spread kernel still sees the norms uploaded for its own call, but
     * the host no longer blocks/wait-blocks the GPU pipeline 32x per
     * chunk batch. */
    cudaMemcpyToSymbolAsync(c_norm_data, norm_data, sizeof(float) * num, 0,
                            cudaMemcpyHostToDevice, 0);
}

// Round a / b to nearest higher integer value
int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }


void sort_arrat_thrust(float *indata, int N)
{
    thrust::device_ptr<float> indata_prt(indata);
    thrust::sort(indata_prt, indata_prt+N);
}


float get_mean_gpu(float *indata, int N)
{
    thrust::device_ptr<float> indata_bk(indata);
    double sum = thrust::reduce(indata_bk, indata_bk+N);
    return sum/N;
}

double get_mean_gpu_d(double *indata, int N)
{
    thrust::device_ptr<double> indata_bk(indata);
    double sum = thrust::reduce(indata_bk, indata_bk+N);
    return sum/N;
}


struct variance: std::unary_function<float, float>
{
    variance(float m): mean(m){ }
    const float mean;
    __host__ __device__ float operator()(float data) const
    {
        return (data - mean)*(data - mean);
    }
};


float get_variance_gpu(float *indata, float mean,int N)
{
    thrust::device_ptr<float> indata_bk(indata);
    float variance_bk = thrust::transform_reduce(indata_bk, indata_bk+N, variance(mean), 0.0f, thrust::plus<float>())/N;
    return variance_bk;
}

void get_maxindex_double_gpu(double *indata, int N, double *maxV, int *maxindex)
{
    thrust::device_ptr<double> indata_bk(indata);
    thrust::device_ptr<double> iter = thrust::max_element(indata_bk, indata_bk+N);
    double maxValue = *iter;
    // return iter - indata_bk;
    *maxV = maxValue;
    *maxindex = iter - indata_bk;
}

void sum_value_gpu(float *indata, float value,int N)
{
    int BlkPerRow=(N-1+1024)/1024;
    Do_sum_value_gpu<<< BlkPerRow, 1024>>>(indata, value, N);
}

void sum_value_gpu_stream(float *indata, double *mean,int N, int M, cudaStream_t stream_1, cudaStream_t stream_2)
{
    int i;
    int BlkPerRow=(N-1+1024)/1024;
    for(i=0; i<M-1; i++)
    {   
        Do_sum_value_gpu<<< BlkPerRow, 1024, 0, stream_1>>>(indata+i*N, -mean[i%2], N);
        i++;
        Do_sum_value_gpu<<< BlkPerRow, 1024, 0, stream_2>>>(indata+i*N, -mean[i%2], N);
    }
    cudaStreamSynchronize(stream_1);
    cudaStreamSynchronize(stream_2);

    for(; i<M; i++)
    {   
        Do_sum_value_gpu<<< BlkPerRow, 1024, 0>>>(indata+i*N, -mean[i%2], N);
    }

}

__global__ void Do_sum_value_gpu(float *indata, float value,int N)
{
    const int MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=N) return;

    indata[MYgtid] += value;
}

//----------------------select a cpu to play with --------------------
void select_cuda_dev(int cuda_inds)
{
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess || current != cuda_inds)
        cudaSetDevice(cuda_inds);
    // Selection is idempotent: do not reset the context here.
}


void Endup_GPU()
{
    cudaDeviceSynchronize();
    // cudaDeviceReset();
}


__device__ void warpReduce(volatile float *sdata, unsigned int tid, int blockSize)
{
    if (blockSize >= 64) {sdata[tid] += sdata[tid + 32];}
    if (blockSize >= 32) {sdata[tid] += sdata[tid + 16];}
    if (blockSize >= 16) {sdata[tid] += sdata[tid + 8];}
    if (blockSize >= 8) {sdata[tid] += sdata[tid + 4];}
    if (blockSize >= 4) {sdata[tid] += sdata[tid + 2];}
    if (blockSize >= 2) {sdata[tid] += sdata[tid + 1];}
}

__device__ void warpReduce_double(volatile double *sdata, unsigned int tid, int blockSize)
{
    if (blockSize >= 64) {sdata[tid] += sdata[tid + 32];}
    if (blockSize >= 32) {sdata[tid] += sdata[tid + 16];}
    if (blockSize >= 16) {sdata[tid] += sdata[tid + 8];}
    if (blockSize >= 8) {sdata[tid] += sdata[tid + 4];}
    if (blockSize >= 4) {sdata[tid] += sdata[tid + 2];}
    if (blockSize >= 2) {sdata[tid] += sdata[tid + 1];}
}


unsigned int calc_next_32_times(unsigned int x)
{
    unsigned int next_pow_32_times = WARP_SIZE;
    for(int i=2; next_pow_32_times < x; i++)
        next_pow_32_times = WARP_SIZE*i;
    return next_pow_32_times;
}


int configure_reduce_sum(dim3 &grid_dim, dim3 &block_dim, unsigned int next_pow_32_times)
{
    if(next_pow_32_times < MAX_THREAD)
    {
        block_dim = dim3(next_pow_32_times, 1, 1);
        grid_dim = dim3(1,1,1);
    }
    else
    {
        int grid_x = ceil((double) next_pow_32_times / MAX_THREAD);
        block_dim = dim3(MAX_THREAD, 1, 1);
        grid_dim = dim3(grid_x,1,1);
    }
    return 0;
}



float gpu_sum_reduce_float(float *arr, int total_num)
{
    static float   *d_out_arr, *h_out_arr; //*d_arr,
    int next_pow_32_times = calc_next_32_times(total_num);
    static int firsttime = -1;
    // cudaMalloc(&d_arr, next_pow_32_times * sizeof(float));
    // cudaMemset(d_arr, 0, next_pow_32_times * sizeof(float));
    
    dim3 block_dim, grid_dim;
    configure_reduce_sum(grid_dim, block_dim, next_pow_32_times);
    
    int block_num = grid_dim.x * grid_dim.y * grid_dim.z;

    
    if(firsttime==-1)
    {
        cudaMalloc(&d_out_arr, block_num * sizeof(float));
        h_out_arr = new float[block_num];

        firsttime = total_num;
    }
    else if(firsttime!=total_num)
    {
        cudaFree(d_out_arr);
        delete []h_out_arr;

        cudaMalloc(&d_out_arr, block_num * sizeof(float));
        h_out_arr = new float[block_num];

        firsttime = total_num;
    }

    
    cudaMemset(d_out_arr, 0, block_num * sizeof(float));

    // cudaMemcpy(d_arr, arr, sizeof(float) * total_num, cudaMemcpyHostToDevice);

    // reduce6<<<grid_dim, block_dim, block_dim.x*sizeof(int)>>>(d_arr, d_out_arr, total_num, block_dim.x);
    block_sum_reduce_float<<<grid_dim, block_dim, block_dim.x*sizeof(float)>>>(arr, d_out_arr, total_num, block_dim.x);
    // cudaDeviceSynchronize();

    
    // memset(h_out_arr, 0, sizeof(float)*block_num);
    cudaMemcpy(h_out_arr, d_out_arr, sizeof(float) * block_num, cudaMemcpyDeviceToHost);

    float ret=0.0f;
    for(int i=0; i<block_num; i++)
    {
        ret += h_out_arr[i];
    }

    // cudaFree(d_arr);

    // {
    //     cudaFree(d_out_arr);
    //     delete []h_out_arr;
    // }
    

    return ret;

}


__global__ void block_sum_reduce_float(float *g_idata, float *g_odata, unsigned int n, int blockSize) 
{
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid; 
    unsigned int gridSize = blockSize*2*gridDim.x; //suit for the grid_dim.y > 1

    sdata[tid] = 0;//set the shared memory to zero is needed
    while (i < n) 
    { 
        sdata[tid] += g_idata[i];
        if ((i+blockSize) < n)
            sdata[tid] += g_idata[i+blockSize];
        
        i += gridSize; 
    }
     __syncthreads();
    if (blockSize >= 1024) 
    { 
        if (tid < 512)
        { 
            sdata[tid] += sdata[tid + 512]; 
        }
        __syncthreads(); 
    } 
    if (blockSize >= 512) 
    { 
        if (tid < 256)
        { 
            sdata[tid] += sdata[tid + 256]; 
        }
        __syncthreads(); 
    } 
    if (blockSize >= 256) 
    { 
        if (tid < 128) 
        { 
            sdata[tid] += sdata[tid + 128]; 
        }
        __syncthreads(); 
    } 
    if (blockSize >= 128)
    { 
        if (tid < 64)
        { 
            sdata[tid] += sdata[tid + 64]; 
        } 
        __syncthreads(); 
    }
    if (tid < 32) warpReduce(sdata, tid, blockSize);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0]; 
}

float gpu_varience_reduce_float(float *arr, float mean, int total_num)
{
    double  *d_out_arr, *h_out_arr; //*d_arr,
    int next_pow_32_times = calc_next_32_times(total_num);

    // cudaMalloc(&d_arr, next_pow_32_times * sizeof(float));
    // cudaMemset(d_arr, 0, next_pow_32_times * sizeof(float));
    
    dim3 block_dim, grid_dim;
    configure_reduce_sum(grid_dim, block_dim, next_pow_32_times);
    
    int block_num = grid_dim.x * grid_dim.y * grid_dim.z;
   
    cudaMalloc(&d_out_arr, block_num * sizeof(double));
    cudaMemset(d_out_arr, 0, block_num * sizeof(double));

    // cudaMemcpy(d_arr, arr, sizeof(float) * total_num, cudaMemcpyHostToDevice);
    
    // reduce6<<<grid_dim, block_dim, block_dim.x*sizeof(int)>>>(d_arr, d_out_arr, total_num, block_dim.x);
    block_varience_reduce_float<<<grid_dim, block_dim, block_dim.x*sizeof(double)>>>(arr, d_out_arr, mean, total_num, block_dim.x);
    // cudaDeviceSynchronize();

    h_out_arr = new double[block_num];
    // memset(h_out_arr, 0, sizeof(double)*block_num);
    cudaMemcpy(h_out_arr, d_out_arr, sizeof(double) * block_num, cudaMemcpyDeviceToHost);

    double ret=0;
    for(int i=0; i<block_num; i++)
    {
        ret += h_out_arr[i];
    }
    // cudaFree(d_arr);
    cudaFree(d_out_arr);
    delete []h_out_arr;
    return (float)ret;
}


__global__ void block_varience_reduce_float(float *g_idata, double *g_odata, float mean, unsigned int n, int blockSize) 
{
    extern __shared__ double ddata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid; 
    unsigned int gridSize = blockSize*2*gridDim.x; //suit for the grid_dim.y > 1

    double bk;
    ddata[tid] = 0;//set the shared memory to zero is needed
    while (i < n) 
    { 
        bk=g_idata[i]-mean;
        ddata[tid] += bk*bk;
        if ((i+blockSize) < n)
        {
            bk=g_idata[i+blockSize]-mean;
            ddata[tid] += bk*bk;
        }
            
        
        i += gridSize; 
    }
     __syncthreads();

    if (blockSize >= 1024) 
    { 
        if (tid < 512)
        { 
            ddata[tid] += ddata[tid + 512]; 
        }
        __syncthreads(); 
    } 
    if (blockSize >= 512) 
    { 
        if (tid < 256)
        { 
            ddata[tid] += ddata[tid + 256]; 
        }
        __syncthreads(); 
    } 
    if (blockSize >= 256) 
    { 
        if (tid < 128) 
        { 
            ddata[tid] += ddata[tid + 128]; 
        }
        __syncthreads(); 
    } 
    if (blockSize >= 128)
    { 
        if (tid < 64)
        { 
            ddata[tid] += ddata[tid + 64]; 
        } 
        __syncthreads(); 
    }
    if (tid < 32) warpReduce_double(ddata, tid, blockSize);
    if (tid == 0) g_odata[blockIdx.x] = ddata[0]; 
}

void mask_data_GPU(float *currentdata_gpu, int *maskchans_gpu, float *padvals_gpu, int spectra_per_subint, int num_channels, int nummasked)
{
    int BlkPerRow=(spectra_per_subint*nummasked-1+512)/512;
    // dim3 dimGrid2D(BlkPerRow,numsubbands);
    Do_mask_data_GPU<<<BlkPerRow, 512>>>(currentdata_gpu, maskchans_gpu, padvals_gpu, spectra_per_subint, num_channels, nummasked);
    // int cudaStatus = cudaDeviceSynchronize();
}

__global__ void Do_mask_data_GPU(float *currentdata_gpu, int *maskchans_gpu, float *padvals_gpu, int spectra_per_subint, int num_channels, int nummasked)
{
    const int MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=spectra_per_subint*nummasked) return;

    int i,j;
    i=MYgtid/nummasked;     //当前point
    j=MYgtid%nummasked;     //当前maskid

    int offset = i*num_channels;
    int channum = maskchans_gpu[j];
    currentdata_gpu[offset+channum] = padvals_gpu[channum];
}

void ignorechans_GPU(float *currentdata_gpu, int *ignorechans_gpu, int spectra_per_subint, int num_channels, int num_ignorechans)
{
    int BlkPerRow=(spectra_per_subint*num_ignorechans-1+512)/512;
    DO_ignorechans_GPU<<<BlkPerRow, 512>>>(currentdata_gpu, ignorechans_gpu, spectra_per_subint, num_channels, num_ignorechans);
    // int cudaStatus = cudaDeviceSynchronize();
}

__global__ void DO_ignorechans_GPU(float *currentdata_gpu, int *ignorechans_gpu, int spectra_per_subint, int num_channels, int num_ignorechans)
{
    const int MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=spectra_per_subint*num_ignorechans) return;

    int i,j;
    i=MYgtid/num_ignorechans;     //当前point
    j=MYgtid%num_ignorechans;     //当前maskid

    int offset = i*num_channels;
    int channum = ignorechans_gpu[j];

    currentdata_gpu[offset + channum] = 0.0f;
}

void dedisp_subbands_GPU(float *data_gpu, float *lastdata_gpu,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose)
{
    // static float *data_gpu, *lastdata_gpu;
    // static int firsttime=1;

    // if(firsttime)
    // {
    //     cudaMalloc((void**)&data_gpu, sizeof(float)*numpts * numchan);
    //     cudaMalloc((void**)&lastdata_gpu, sizeof(float)*numpts * numchan);
    //     firsttime=0;
    // }

    // cudaMemcpy(data_gpu, data, sizeof(float)*numpts * numchan, cudaMemcpyHostToDevice);
    // cudaMemcpy(lastdata_gpu, lastdata, sizeof(float)*numpts * numchan, cudaMemcpyHostToDevice);
    

    int BlkPerRow=(numpts*numsubbands-1+512)/512;
    // dim3 dimGrid2D(BlkPerRow,numsubbands);
    Do_dedisp_subbands_GPU<<<BlkPerRow,512>>>(data_gpu, lastdata_gpu, numpts, numchan, delays, numsubbands, result, transpose);
    // int cudaStatus = cudaDeviceSynchronize();

    // cudaFree(data_gpu);
    // cudaFree(lastdata_gpu);
}

void dedisp_subbands_GPU_cache_stream(unsigned char *data_gpu, float *data_gpu_scl, float *data_gpu_offs, unsigned char *lastdata_gpu, float *lastdata_gpu_scl, float *lastdata_gpu_offs,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose, cudaStream_t stream)
{
    int BlkPerRow=(numpts*numsubbands-1+512)/512;
    Do_dedisp_subbands_GPU_cache<<<BlkPerRow, 512, 0, stream>>>(data_gpu, data_gpu_scl, data_gpu_offs, lastdata_gpu, lastdata_gpu_scl, lastdata_gpu_offs,
                                 numpts, numchan, delays, numsubbands, result, transpose);
}

void dedisp_subbands_GPU_cache(unsigned char *data_gpu, float *data_gpu_scl, float *data_gpu_offs, unsigned char *lastdata_gpu, float *lastdata_gpu_scl, float *lastdata_gpu_offs,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose)
{
    int BlkPerRow=(numpts*numsubbands-1+512)/512;
    // dim3 dimGrid2D(BlkPerRow,numsubbands);
    Do_dedisp_subbands_GPU_cache<<<BlkPerRow,512>>>(data_gpu, data_gpu_scl, data_gpu_offs, lastdata_gpu, lastdata_gpu_scl, lastdata_gpu_offs,
                                 numpts, numchan, delays, numsubbands, result, transpose);

    // int BlkPerRow=(numpts*numsubbands-1+512)/512;
    // Do_dedisp_subbands_GPU_cache<<<BlkPerRow,512>>>(data_gpu, data_gpu_scl, data_gpu_offs, lastdata_gpu, lastdata_gpu_scl, lastdata_gpu_offs,
    //                              numpts, numchan, delays, numsubbands, result, transpose);

    // int cudaStatus = cudaDeviceSynchronize();
}

__global__ void Do_dedisp_subbands_GPU(float *data, float *lastdata,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose)
{
    // int i;
    // const int MYrow=blockIdx.y;
    // const int MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    // if(MYgtid>=numpts) return;

    const int alltid = blockDim.x * blockIdx.x + threadIdx.x;
    if(alltid>=numpts*numsubbands)  return;
    int i;
    const int MYrow = alltid/numpts;
    const int MYgtid = alltid%numpts;


    float temp = 0.0f;
    int chan_per_subband = numchan / numsubbands;
    int  in_index, out_index;
    if(transpose)
        out_index = MYrow*numpts + MYgtid;      // transpose=1 for time first
    else
        out_index = MYrow + MYgtid*numsubbands;  // transpose=0 for freq first
    int blk = numpts*numchan;
    for(i=0;i<chan_per_subband; i++)
    {
        int  in_chan = i + MYrow*chan_per_subband;
        int  dind = delays[in_chan];
        in_index = (dind+MYgtid)*numchan + in_chan;
        if(MYgtid < numpts-dind)
        {
            temp += lastdata[in_index];
        }
        else
        {
            in_index -= blk;
            temp += data[in_index];
        }
    }
    result[out_index] = temp;
}

__global__ void Do_dedisp_subbands_GPU_cache(unsigned char *data, float *data_scl, float *data_offs, unsigned char *lastdata, float *lastdata_scl, float *lastdata_offs,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose)
{
    // int i;
    // const int MYrow=blockIdx.y;
    // const int MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    // if(MYgtid>=numpts) return;

    const int alltid = blockDim.x * blockIdx.x + threadIdx.x;
    if(alltid>=numpts*numsubbands)  return;
    int i;
    const int MYrow = alltid/numpts;
    const int MYgtid = alltid%numpts;

    float temp = 0.0f;
    int chan_per_subband = numchan / numsubbands;
    int  in_index, out_index;
    if(transpose)
        out_index = MYrow*numpts + MYgtid;      // transpose=1 for time first
    else
        out_index = MYrow + MYgtid*numsubbands;  // transpose=0 for freq first
    
    for(i=0;i<chan_per_subband; i++)
    {
        int  in_chan = i + MYrow*chan_per_subband;
        int  dind = delays[in_chan];
        in_index = dind+MYgtid + in_chan*numpts;
        if((MYgtid + dind) < numpts)
        {
            temp += (lastdata[in_index]*lastdata_scl[in_chan]+lastdata_offs[in_chan]);
        }
        else
        {
            in_index -= numpts;
            temp += (data[in_index]*data_scl[in_chan]+data_offs[in_chan]);
        }
    }
    result[out_index] = temp;
}

void downsamp_GPU(float *indata, float *outdata, int numchan, int numpts, int down, int transpose)
{
    int BlkPerRow=(numpts*numchan-1+512)/512;
    // dim3 dimGrid2D(BlkPerRow,numchan);
    Do_downsamp_GPU<<<BlkPerRow, 512>>>(indata, outdata, numchan, numpts, down, transpose);
    // int cudaStatus = cudaDeviceSynchronize();
}

__global__ void Do_downsamp_GPU(float *indata, float *outdata, int numchan, int numpts, int down, int transpose)
{
    // int i;
    // const int MYrow=blockIdx.y;
    // const int MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    // if(MYgtid>=numpts) return;

    const int alltid = blockDim.x * blockIdx.x + threadIdx.x;
    if(alltid>=numpts*numchan)  return;
    int i;
    const int MYrow = alltid/numpts;
    const int MYgtid = alltid%numpts;

    float ftmp = 0.0f;
    
    int  in_dex;

    if(transpose)
    {
        /* input time first */
        const int out_index = MYrow*numpts + MYgtid;
        in_dex = out_index*down;
        for(i=0;i<down;i++)
        {
            ftmp+=indata[in_dex];
            in_dex++;
        }
        outdata[out_index] = ftmp/(1.0f*down);
    }
    else
    {
        // /* input freq first */
        const int out_index = MYrow + MYgtid*numchan;
        in_dex = MYgtid*numchan*down + MYrow;
        for(i=0;i<down;i++)
        {
            ftmp+=indata[in_dex];
            in_dex+=numchan;
        }
        outdata[out_index] = ftmp /(1.0f*down);
    }
}

void float_dedisp_GPU(float *data, float *lastdata,
                  int numpts, int numchan,
                  int *delays, float approx_mean, float *result, int numdms, int transpose)
{

    int BlkPerRow=(numpts*numdms-1+512)/512;
    // dim3 dimGrid2D(BlkPerRow,numdms);
    Do_float_dedisp_GPU<<<BlkPerRow,512>>>(data, lastdata, numpts, numchan,
                  delays, approx_mean, result, numdms, transpose);
    // int cudaStatus = cudaDeviceSynchronize();
}


__global__ void Do_float_dedisp_GPU(float *data, float *lastdata,
                  int numpts, int numchan,
                  int *delays, float approx_mean, float *result, int numdms, int transpose)
{
    // const int MYrow=blockIdx.y;
    // const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    // if(MYgtid>=numpts) return;
    const int alltid = blockDim.x * blockIdx.x + threadIdx.x;
    if(alltid>=numpts*numdms) return;
    const int MYrow = alltid/numpts;
    const int MYgtid = alltid%numpts;
    int i;

    
    int in_dex;
    const int out_index = MYrow*numpts + MYgtid;
    const int  x=MYrow*numchan;

    float ftmp = -approx_mean;

    


    /* input time first */
    if(transpose)
    {
        for(i=0;i<numchan;i++)
        {
            in_dex = MYgtid + delays[x+i];
            if(in_dex < numpts)
            {
                in_dex = in_dex + i*numpts;
                ftmp += lastdata[in_dex];
            }
            else 
            {
                in_dex = in_dex - numpts + i*numpts;
                ftmp += data[in_dex];
            }
        }
        result[out_index] = ftmp;
    }
    /* input freq first */
    else
    {
        for(i=0;i<numchan;i++)
        {
            in_dex = MYgtid+(long long)(delays[x+i]);
            if(in_dex<numpts)
            {
                in_dex = i + in_dex*numchan;
                ftmp += lastdata[in_dex];
            }
            else
            {
                in_dex -= numpts;
                in_dex = i + in_dex*numchan;
                ftmp += data[in_dex];
            }
        }
        result[out_index] = ftmp;
    }
    
}


void Get_subsdata(float *indata, short *outdata, int nsub, int worklen)
{
    int BlkPerRow=(nsub*worklen-1+512)/512;
    Do_Get_subsdata<<< BlkPerRow,512 >>>(indata, outdata, nsub, worklen);
    // int cudaStatus = cudaDeviceSynchronize();
}

__global__ void Do_Get_subsdata(float *indata, short *outdata, int nsub, int worklen)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=worklen) return;

    int x,y;
    x = MYgtid%worklen;
    y = MYgtid/worklen;
    outdata[MYgtid] = (short)(indata[x*nsub+y]+0.5);
}


void SetSubData4Mask_GPU(float *subbanddata, float *padvals, int *maskchans,int len, int Width)
{
    int BlkPerRow=(len-1+512)/512;
    dim3 dimGrid2D(BlkPerRow,Width);
    Do_SetSubData4Mask_GPU<<< dimGrid2D,512 >>>(subbanddata, padvals, maskchans, len, Width);
    // int cudaStatus = cudaDeviceSynchronize();
}

__global__ void Do_SetSubData4Mask_GPU(float *subbanddata, float *padvals, int *maskchans, int len, int Width)
{
    const int MYrow=blockIdx.y;
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=len) return;

    int channum = maskchans[MYrow];
    int in_dex = MYgtid*Width+channum;
    subbanddata[in_dex] = padvals[channum];
}


void ZeroDM_subchan_GPU(float *indata, int len, int Width)
{
    int ii;
    float ave;
    int BlkPerRow;
    // int cudaStatus;

    for(ii=0; ii< len; ii++)
    {
        ave=gpu_sum_reduce_float(indata+ii*Width,Width);
        ave/=Width;
        BlkPerRow=(Width-1+512)/512;
        Add_data_GPU<<<BlkPerRow,512>>>(indata, -ave, Width);
        // cudaStatus = cudaDeviceSynchronize();
    }

}

__global__ void Add_data_GPU(float *indata, float value, int len)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=len) return;

    indata[MYgtid] +=  value;
}


void transpose_GPU(float *d_src, float *d_dest, int width, int height) 
{
  int BlkPerRow = (width*height-1+512)/512;
  d_transpose<<<BlkPerRow, 512>>>(d_dest, d_src, width, height);
//   int cudaStatus =  cudaDeviceSynchronize();
}

__global__ void d_transpose(float *odata, float *idata, int width, int height) 
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=(width*height)) return;

    // int yIndex = MYgtid/width;
    // int xIndex = MYgtid-yIndex*width;
    // int index_in = yIndex + xIndex * height;
    // odata[MYgtid] = idata[index_in];

    int yIndex = MYgtid%height;
    int xIndex = MYgtid/height;
    int index_out = yIndex*width + xIndex;

    odata[index_out] = idata[MYgtid];
}




void transpose_GPU_short(short *d_src, float *d_dest, int width, int height) 
{
  int BlkPerRow =(height-1+512)/512;
  d_transpose_short<<<BlkPerRow*width, 512>>>(d_dest, d_src, width, height);

//   dim3 grid(iDivUp(width, BLOCK_DIM), iDivUp(height, BLOCK_DIM), 1);
//   dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
//   d_transpose<<<grid, threads>>>(d_dest, d_src, width, height);

//   int cudaStatus =  cudaDeviceSynchronize();
}

__global__ void d_transpose_short(float *odata, short *idata, int width, int height) 
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=(width*height)) return;

    int yIndex = MYgtid/width;
    int xIndex = MYgtid-yIndex*width;

    int index_in = yIndex + xIndex * height;
	odata[MYgtid] = idata[index_in];
}

void FFTonGPU(cufftComplex *GPUTimeDMArray,cufftComplex *GPUTimeDMArrayFFTbk, int x,int y, int index, cufftHandle plan)
{
    /*方法一*/
    // cufftPlan1d(&plan, x, CUFFT_C2C, y);
    cufftExecC2C(plan, GPUTimeDMArray, GPUTimeDMArrayFFTbk,index);
    // int cudaStatus = cudaDeviceSynchronize();
    // cufftDestroy(plan);
}

void get_power_GPU(cufftComplex * data, int numdata, float *power)
{
    int BlkPerRow = (numdata-1+512)/512;
    Do_get_power_GPU<<<BlkPerRow, 512>>>(data, numdata, power);
}

__global__ void Do_get_power_GPU(cufftComplex * data, int numdata, float *power)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numdata) return;

    cufftComplex data_bk = data[MYgtid];
    power[MYgtid] = data_bk.x* data_bk.x + data_bk.y*data_bk.y;
}

float  get_med_gpu(float *data, int N)
{
    sort_arrat_thrust(data, N);
    float med;
    cudaMemcpy(&med, &data+N/2, sizeof(float), cudaMemcpyDeviceToHost);
    return med;
}


void spread_no_pad_gpu(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, float norm)
{
    int BlkPerRow = (numdata-1+512)/512;
    Do_spread_with_pad_GPU<<<BlkPerRow, 512>>>(data, numdata, result, numresult, numbetween, 0,norm);
    // CUDA_CHECK(cudaGetLastError());
}

__global__ void Do_spread_with_pad_GPU(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, int numpad, float norm)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    // if(MYgtid>=numresult) return;

    // cufftComplex zeros = { 0.0f, 0.0f};
    // result[MYgtid] = zeros;
    
    if(MYgtid >= numdata) return;
    cufftComplex data_bk = data[MYgtid];
    data_bk.x *= norm;
    data_bk.y *= norm;
    
    result[MYgtid*numbetween] = data_bk;
}

void spread_no_pad_gpu_dat(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, float norm, long long offset_bk, long long newnumbins_bk)
{
    // int BlkPerRow = (numresult-offset_bk-1+256)/256;
    int BlkPerRow = (numresult-1+256)/256;
    Do_spread_with_pad_GPU_dat<<<BlkPerRow, 256>>>(data, numdata, result, numresult, numbetween, 0, norm, offset_bk, newnumbins_bk);
    // CUDA_CHECK(cudaGetLastError());
}

__global__ void Do_spread_with_pad_GPU_dat(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, int numpad, float norm, long long offset_bk, long long newnumbins_bk)
{
    // const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    // if(MYgtid>=numresult-offset_bk) return;
    // const float data_bk_x =  __ldg(&data[MYgtid].x) * norm;
    // const float data_bk_y =  __ldg(&data[MYgtid].y) * norm;
    // result[(MYgtid+offset_bk)*numbetween] = {data_bk_x, data_bk_y};
    

    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numresult) return;

    if(MYgtid<offset_bk)
    {
        result[MYgtid] = {0.0f, 0.0f};
    }
    else 
    {
        float2 data_bk = __ldg(&data[MYgtid]);
        const float data_bk_x =  data_bk.x * norm;
        const float data_bk_y =  data_bk.y * norm;
        result[MYgtid*numbetween] = {data_bk_x, data_bk_y};
        result[MYgtid*numbetween + 1] = {0.0f, 0.0f};
    }
}

void spread_no_pad_gpu_list(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, int readdatanum, float *norm_data_gpu)
{
    int BlkPerRow = (numdata*readdatanum-1+256)/256;
    Do_spread_with_pad_GPU_list<<<BlkPerRow, 256>>>(data, numdata, result, numresult, numbetween, 0, readdatanum, norm_data_gpu);
    // CUDA_CHECK(cudaGetLastError());
}

__global__ void Do_spread_with_pad_GPU_list(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, int numpad, int readdatanum, float *norm_data_gpu)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    // if(MYgtid>=numresult*readdatanum) return;

    // cufftComplex zeros = { 0.0f, 0.0f};
    // result[MYgtid] = zeros;

    if(MYgtid>=numdata*readdatanum) return;

    // int id_x = MYgtid%numdata;
    int id_y = MYgtid/numdata;

    cufftComplex data_bk = __ldg(&data[MYgtid]);
    // float norm_bk = __ldg(&norm_data_gpu[id_y]);
    float norm_bk = c_norm_data[id_y];

    data_bk.x *= norm_bk;
    data_bk.y *= norm_bk;
    
    result[MYgtid*numbetween] = data_bk;
}

// void spread_no_pad_gpu_dat_list_loiops(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, float *norm, long long offset_bk, long long numpad_bk, long long newnumbins_bk, long long addbins, int readdatanum, int data_len)
// {
//     int BlkPerRow = (numdata*readdatanum-1+256)/256;
//     Do_spread_with_pad_GPU_dat_list_loiops<<<BlkPerRow, 256>>>(data, numdata, result, numresult, numbetween, 0, norm, offset_bk, numpad_bk, newnumbins_bk, addbins, readdatanum, data_len);
// }


// __global__ void Do_spread_with_pad_GPU_dat_list_loiops(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, int numpad, float *norm, long long offset_bk, long long numpad_bk, long long newnumbins_bk, long long addbins, int readdatanum, int data_len)
// {
//     const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
//     if(MYgtid>=numdata*readdatanum) return;

//     int id_x = MYgtid%numdata;
//     int id_y = MYgtid/numdata;

//     // float norm_bk = __ldg(&norm[id_y]);
//     float norm_bk = c_norm_data[id_y];

//     float2 data_bk = __ldg(&data[id_y*data_len + id_x + addbins]);

//     const float data_bk_x =  data_bk.x * norm_bk;
//     const float data_bk_y =  data_bk.y * norm_bk;
//     result[MYgtid*numbetween] = {data_bk_x, data_bk_y};
// }

void spread_no_pad_gpu_dat_list_loiops(cufftComplex * data, int numdata, cufftComplex *result, int numresult, int numbetween, float *norm, long long offset_bk, long long numpad_bk, long long newnumbins_bk, long long addbins, int readdatanum, int data_len)
{
    int BlkPerRow = (numresult*readdatanum-1+256)/256;
    Do_spread_with_pad_GPU_dat_list_loiops<<<BlkPerRow, 256>>>(data, numdata, result, numresult, numbetween, 0, norm, offset_bk, numpad_bk, newnumbins_bk, addbins, readdatanum, data_len);
}


__global__ void Do_spread_with_pad_GPU_dat_list_loiops(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, int numpad, float *norm, long long offset_bk, long long numpad_bk, long long newnumbins_bk, long long addbins, int readdatanum, int data_len)
{
    int MYgtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (MYgtid >= numresult * readdatanum) return;

    int k = MYgtid / numbetween;                 // 对应原来的 MYgtid
    if (MYgtid % numbetween == 0 && k < numdata * readdatanum) {
        // 需要写入数据的位置
        int id_x = k % numdata;
        int id_y = k / numdata;
        // float norm_bk = __ldg(&norm[id_y]);                     // 归一化系数
        float norm_bk = c_norm_data[id_y];
        float2 data_bk = __ldg(&data[id_y * data_len + id_x + addbins]);
        result[MYgtid].x = data_bk.x * norm_bk;
        result[MYgtid].y = data_bk.y * norm_bk;
    } else {
        // 其余位置置零
        result[MYgtid] = {0.0f, 0.0f};
        return;
    }
}


void loops_in_GPU_1(cufftComplex *fpdata, cufftComplex *fkern, cufftComplex *outdata, int fftlen, int numzs)
{
    /*********************************** 方式 0： *********************************************************/
    int BlkPerRow = (fftlen*numzs-1+256)/256;
    Do_loops_in_GPU_1<<<BlkPerRow, 256>>>(fpdata, fkern, outdata, fftlen, numzs);
}

static __global__ void Do_loops_in_GPU_1(cufftComplex *fpdata, cufftComplex *fkern, cufftComplex *outdata, int fftlen, int numzs)
{

    /*********************************** 方式 0： *********************************************************/
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=fftlen*numzs) return;
    const int xnum = MYgtid%fftlen;

    /*********************************************************************/
    // const float dr = __ldg(&fpdata[xnum].x);
    // const float di = __ldg(&fpdata[xnum].y);
    // const float kr = __ldg(&fkern[MYgtid].x);
    // const float ki = __ldg(&fkern[MYgtid].y);
    
    // cufftComplex outdata_bk;
    // outdata_bk.x = fmaf(dr, kr, di * ki);  // fused multiply-add
    // outdata_bk.y = fmaf(di, kr, -dr * ki);
    // outdata[MYgtid] = outdata_bk;


    /*********************************************************************/
    // const float dr = fpdata[xnum].x, di = fpdata[xnum].y;
    // const float kr = fkern[MYgtid].x, ki = fkern[MYgtid].y;
    // cufftComplex outdata_bk;
    // outdata_bk.x = dr * kr + di * ki;
    // outdata_bk.y = di * kr - dr * ki;
    // outdata[MYgtid] = outdata_bk;

    /*********************************************************************/
    const  float2 fdata_d = __ldg(&fpdata[xnum]);
    const  float2 fkern_k = __ldg(&fkern[MYgtid]);

    cufftComplex outdata_bk;
    outdata_bk.x = fdata_d.x * fkern_k.x + fdata_d.y * fkern_k.y;
    outdata_bk.y = fdata_d.y * fkern_k.x - fdata_d.x * fkern_k.y;
    outdata[MYgtid] = outdata_bk;
}

// void loops_in_GPU_1(cufftComplex *fpdata, cufftComplex *fkern,
//                     cufftComplex *outdata, int fftlen, int numzs)
// {
//     int threads = 256;
//     int blocks = (fftlen + threads - 1) / threads;
//     Do_loops_in_GPU_1<<<blocks, threads>>>(fpdata, fkern, outdata, fftlen, numzs);
// }

// static __global__ void Do_loops_in_GPU_1(cufftComplex *fpdata,
//                                          cufftComplex *fkern,
//                                          cufftComplex *outdata,
//                                          int fftlen, int numzs)
// {
//     int xnum = blockIdx.x * blockDim.x + threadIdx.x;
//     if (xnum >= fftlen) return;

//     float2 fdata = __ldg(&fpdata[xnum]);   // 每个 x 只读一次

//     for (int z = 0; z < numzs; ++z) {
//         int idx = z * fftlen + xnum;
//         float2 fkern_k = __ldg(&fkern[idx]);

//         float2 out;
//         out.x = fdata.x * fkern_k.x + fdata.y * fkern_k.y;
//         out.y = fdata.y * fkern_k.x - fdata.x * fkern_k.y;
//         outdata[idx] = out;
//     }
// }

void loops_in_GPU_1_list(cufftComplex *fpdata, cufftComplex *fkern, cufftComplex *outdata, int fftlen, int numzs, int readdatanum)
{
    int BlkPerRow = (fftlen*numzs*readdatanum-1+256)/256;
    Do_loops_in_GPU_1_list<<<BlkPerRow, 256>>>(fpdata, fkern, outdata, fftlen, numzs,readdatanum);
}

static __global__ void Do_loops_in_GPU_1_list(cufftComplex *fpdata, cufftComplex *fkern, cufftComplex *outdata, int fftlen, int numzs, int readdatanum)
{
    // const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    // if(MYgtid>=fftlen*numzs*readdatanum) return;

    // const int dataid = MYgtid/(fftlen*numzs);
    // const int zid = (MYgtid - dataid*fftlen*numzs)/fftlen;
    // const int xnum = MYgtid%fftlen;
    
    // float2 fpdata_b = __ldg(reinterpret_cast<float2*>(&fpdata[xnum+dataid*fftlen]));
    // float2 fkern_b = __ldg(reinterpret_cast<float2*>(&fkern[xnum+zid*fftlen]));
    // // const float dr = __ldg(&fpdata[xnum+dataid*fftlen].x), di = __ldg(&fpdata[xnum+dataid*fftlen].y);
    // // const float kr = __ldg(&fkern[xnum+zid*fftlen].x), ki = __ldg(&fkern[xnum+zid*fftlen].y);


    // cufftComplex outdata_bk;
    // // outdata_bk.x = dr * kr + di * ki;
    // // outdata_bk.y = di * kr - dr * ki;
    // outdata_bk.x =  fmaf(fpdata_b.x, fkern_b.x, fpdata_b.y*fkern_b.y) ;//fmaf(dr, kr, di * ki);  // fused multiply-add
    // outdata_bk.y =  fmaf(fpdata_b.y, fkern_b.x, -fpdata_b.x*fkern_b.y); // fmaf(di, kr, -dr * ki);

    // outdata[MYgtid] = outdata_bk;


    const int MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if (MYgtid >= fftlen * numzs * readdatanum)
        return;


    int dataid, zid, xnum;
    // float2 fpdata_b, fkern_b;
    // cufftComplex outdata_bk;

    dataid = MYgtid / (fftlen * numzs);
    zid = (MYgtid - dataid * fftlen * numzs) / fftlen;
    xnum = MYgtid % fftlen;

    // fpdata_b = __ldg(reinterpret_cast<float2 *>(&fpdata[xnum + dataid * fftlen]));
    // fkern_b = __ldg(reinterpret_cast<float2 *>(&fkern[xnum + zid * fftlen]));

    // const float dr = __ldg(&fpdata[xnum + dataid * fftlen].x);
    // const float di = __ldg(&fpdata[xnum + dataid * fftlen].y);
    // const float kr = __ldg(&fkern[xnum + zid * fftlen].x);
    // const float ki = __ldg(&fkern[xnum + zid * fftlen].y);

    // cufftComplex outdata_bk;
    // outdata_bk.x = fmaf(dr, kr, di * ki);
    // outdata_bk.y = fmaf(di, kr, -dr * ki);
    // outdata[MYgtid] = outdata_bk;

    const float dr = fpdata[xnum + dataid * fftlen].x, di = fpdata[xnum + dataid * fftlen].y;
    const float kr = fkern[xnum + zid * fftlen].x, ki = fkern[xnum + zid * fftlen].y;
    cufftComplex outdata_bk;
    outdata_bk.x = dr * kr + di * ki;
    outdata_bk.y = di * kr - dr * ki;
    outdata[MYgtid] = outdata_bk;
}

void loops_in_GPU_2(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip)
{
    /*********************************** 方式 0： *********************************************************/
    int BlkPerRow = (numrs*numzs-1+512)/512;
    Do_loops_in_GPU_2<<<BlkPerRow, 512>>>(fdata, outpows, numrs, numzs, offset, fftlen, norm, outpows_obs, rlen, rlo, tip);
}

static __global__ void Do_loops_in_GPU_2(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip)
{
    /*********************************** 方式 0： *********************************************************/
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numrs*numzs) return;

    const int ynum = MYgtid/numrs;
    const int xnum = MYgtid - ynum*numrs;
    const int ind = xnum+offset + ynum*fftlen;

    // Complex fdata_b = fdata[ind];
    const float fr = __ldg(&fdata[ind].x);
    const float fi = __ldg(&fdata[ind].y);
    float real_sq = __fmul_rn(fr, fr);
    float imag_sq = __fmul_rn(fi, fi);
    float power = __fmul_rn(__fadd_rn(real_sq, imag_sq), norm);

    if(tip)
        outpows[MYgtid] = outpows_obs[ynum*rlen+xnum+rlo] =  fmaf(fr, fr, fi*fi)* norm; //(fdata_b.x*fdata_b.x +  fdata_b.y*fdata_b.y) * norm;
    else
        outpows[MYgtid] = power; //fmaf(fr, fr, fi*fi)* norm;

    // if(tip)
    //     outpows[MYgtid] = outpows_obs[ynum*rlen+xnum+rlo] = (fdata_b.x*fdata_b.x +  fdata_b.y*fdata_b.y) * norm;
    // else
    //     outpows[MYgtid] = (fdata_b.x*fdata_b.x + fdata_b.y*fdata_b.y) * norm;
    
}

void loops_in_GPU_2_end(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip, accel_cand_gpu *cand_gpu_cpu, accel_cand_gpu * cand_array_search_gpu, int *nof_cand, float powcut, int numharm, int *cand_num_gpu)
{
    /*********************************** 方式 0： *********************************************************/
    // static int  *d_addr;
    // static int firsttime = 1;

    // if(firsttime)
    // {
    //     cudaMalloc((void **)&d_addr, sizeof(int));
    //     firsttime = 0;
    // }
    // cudaMemset(d_addr, 0, sizeof(int)); // set d_addr to 0

    int BlkPerRow = (numrs*numzs-1+256)/256;
    Do_loops_in_GPU_2_end<<<BlkPerRow, 256>>>(fdata, outpows, numrs, numzs, offset, fftlen, norm, outpows_obs, rlen, rlo, tip, cand_num_gpu, cand_array_search_gpu, powcut, numharm);

    // cudaMemcpy(nof_cand, d_addr, sizeof(int), cudaMemcpyDeviceToHost);

    // if(nof_cand[0])
    //     cudaMemcpy(cand_gpu_cpu, cand_array_search_gpu, sizeof(accel_cand_gpu)* nof_cand[0], cudaMemcpyDeviceToHost);

}


static __global__ void Do_loops_in_GPU_2_end(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip, int *d_addr, accel_cand_gpu * cand_array_search_gpu, float powcut, int numharm)
{
    /*********************************** 方式 0： *********************************************************/
    const long long  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numrs*numzs) return;

    const  long long ynum = MYgtid/numrs;
    const  long long xnum = MYgtid - ynum*numrs;
    const  long long ind = xnum+offset + ynum*fftlen;

    // Complex fdata_b = fdata[ind];


    // 使用__fmul_rn和__fadd_rn提高精度控制
    // const float fr = __ldg(&fdata[ind].x);
    // const float fi = __ldg(&fdata[ind].y);
    // float real_sq = __fmul_rn(fr, fr);
    // float imag_sq = __fmul_rn(fi, fi);
    // float power = __fmul_rn(__fadd_rn(real_sq, imag_sq), norm);

    const  float2 fdata_d = __ldg(&fdata[ind]);
    float real_sq = __fmul_rn(fdata_d.x, fdata_d.x);
    float imag_sq = __fmul_rn(fdata_d.y, fdata_d.y);
    float power = __fmul_rn(__fadd_rn(real_sq, imag_sq), norm);

    // float power = fmaf(fr, fr, fi*fi)* norm; //(fr*fr +  fi*fi) * norm;//__fmul_rn(__fadd_rn(real_sq, imag_sq), norm);

    outpows[MYgtid] = power;
    
    if(tip) {
        outpows_obs[ynum*rlen+xnum+rlo] = power;

        if(power > powcut)
        {
            int addr_search;

            addr_search = atomicAdd(&d_addr[0], 1);

            accel_cand_gpu cand_tmp ;
            cand_tmp.pow = power ;
            cand_tmp.numharm = numharm;
            // cand_tmp.nof_cand = 1 ;
            cand_tmp.z_ind = ynum;
            cand_tmp.r_ind = xnum;
            // cand_tmp.w_ind = 0;
            cand_array_search_gpu[ addr_search ] = cand_tmp ;
        }
    }
    
    

    // if(tip)
    //     outpows[MYgtid] = outpows_obs[ynum*rlen+xnum+rlo] = (fdata_b.x*fdata_b.x +  fdata_b.y*fdata_b.y) * norm;
    // else
    //     outpows[MYgtid] = (fdata_b.x*fdata_b.x + fdata_b.y*fdata_b.y) * norm;
    
}



void loops_in_GPU_2_list(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip, int readdatanum, int outpows_gpu_xlen, long long outpows_gpu_obs_xlen, int *cand_num_gpu, accel_cand_gpu * cand_array_search_gpu, float powcut, int numharm, int output_x_max, int obs_fftlen)
{
    // int BlkPerRow = (numrs*numzs-1+256)/256;
    // dim3 dimGrid2D(BlkPerRow,readdatanum);

    int BlkPerRow = (numrs*numzs*readdatanum-1+256)/256;
    Do_loops_in_GPU_2_list<<<BlkPerRow, 256>>>(fdata, outpows, numrs, numzs, offset, fftlen, norm, outpows_obs, rlen, rlo, tip, readdatanum, outpows_gpu_xlen, outpows_gpu_obs_xlen, cand_num_gpu, cand_array_search_gpu, powcut, numharm, output_x_max, obs_fftlen);
}

static __global__ void Do_loops_in_GPU_2_list(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip, int readdatanum, int outpows_gpu_xlen, long long outpows_gpu_obs_xlen, int *d_addr, accel_cand_gpu * cand_array_search_gpu, float powcut, int numharm, int output_x_max, int obs_fftlen)
{
    const long long  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numrs*numzs*readdatanum) return;
    
    const long long dataid = MYgtid/(numrs*numzs);
    const long long ynum = (MYgtid-dataid*numrs*numzs)/numrs;
    const long long xnum = MYgtid -dataid*numrs*numzs -ynum*numrs;
    
    const long long ind = dataid*fftlen*numzs + ynum*fftlen + xnum + offset;

    // 使用快速数学函数
    const float fr = __ldg(&fdata[ind].x);
    const float fi = __ldg(&fdata[ind].y);
    float real_sq = __fmul_rn(fr, fr);
    float imag_sq = __fmul_rn(fi, fi);
    
    float power = __fmul_rn(__fadd_rn(real_sq, imag_sq), norm);
    outpows[dataid * outpows_gpu_xlen + ynum * numrs + xnum] = power;
    // outpows[dataid * outpows_gpu_xlen + ynum * obs_fftlen + xnum] = power;


    if (tip) {
        outpows_obs[dataid * outpows_gpu_obs_xlen + ynum * rlen + xnum + rlo] = power;

        if(power > powcut)
        {
            int addr_search;

            addr_search = atomicAdd(&d_addr[dataid], 1);

            accel_cand_gpu cand_tmp ;
            cand_tmp.pow = power ;
            cand_tmp.numharm = numharm;
            // cand_tmp.nof_cand = 1 ;
            cand_tmp.z_ind = ynum;
            cand_tmp.r_ind = xnum ;
            // cand_tmp.w_ind = 0;
            cand_array_search_gpu[ addr_search + dataid*output_x_max] = cand_tmp ;
        }
    }
    

}

void add_subharm_gpu(float *powers_out, cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int numzs, int offset, float norm)
{
    /*********************************** 方式 0： *********************************************************/
    int BlkPerRow = (numrs_0*numzs_0-1+256)/256;
    Do_add_subharm_gpu<<< BlkPerRow, 256 >>>(powers_out, fdata, rinds, zinds, numrs_0, numzs_0, fftlen, offset, norm);


    // dim3 block(32,8);
    // dim3 grid((numrs_0+31)/32, (numzs_0+7)/8);
    // Do_add_subharm_gpu_fast<<<grid,block>>>(powers_out, fdata, rinds, zinds, numrs_0, numzs_0, fftlen, offset, norm);
}

static __global__ void Do_add_subharm_gpu(float *powers_out, cufftComplex *fdata,unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int offset, float norm)
{

    /*********************************** 方式 0 ： *********************************************************/
    const long long  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numrs_0*numzs_0) return;

    const long long yy = MYgtid/numrs_0 ;  // numzs_0_id
    unsigned short addr_z = __ldg(&zinds[yy]);
    int xx = MYgtid -  yy * numrs_0 ; // numrs_0_id
    unsigned short addr_r = __ldg(&rinds[xx]);

    Complex fdata_b = fdata[addr_z * fftlen + addr_r + offset];
    // atomicAdd(&powers_out[MYgtid], ((fdata_b.x*fdata_b.x +  fdata_b.y*fdata_b.y)*norm));
    powers_out[MYgtid] += ((fdata_b.x*fdata_b.x +  fdata_b.y*fdata_b.y)*norm);
}

__global__ void Do_add_subharm_gpu_fast(
    float *powers_out,
    cufftComplex *fdata,
    unsigned short *rinds,
    unsigned short *zinds,
    int numrs,
    int numzs,
    int fftlen,
    int offset,
    float norm)
{
    const int xx = blockIdx.x * blockDim.x + threadIdx.x;
    const int yy = blockIdx.y * blockDim.y + threadIdx.y;

    if (xx >= numrs || yy >= numzs) return;

    const int MYgtid = yy * numrs + xx;

    __shared__ unsigned short r_shared[32];
    __shared__ unsigned short z_shared[8];

    if (threadIdx.y == 0)
        r_shared[threadIdx.x] = rinds[xx];

    if (threadIdx.x == 0)
        z_shared[threadIdx.y] = zinds[yy];

    __syncthreads();

    unsigned short addr_r = r_shared[threadIdx.x];
    unsigned short addr_z = z_shared[threadIdx.y];

    cufftComplex fdata_b =
        fdata[addr_z * fftlen + addr_r + offset];

    float power =
        (fdata_b.x * fdata_b.x +
         fdata_b.y * fdata_b.y) * norm;

    powers_out[MYgtid] += power;
}


void add_subharm_gpu_norz(float *powers_out, cufftComplex *fdata, int numrs_0, int numzs_0, int fftlen, int numzs, int offset, float norm, int numharm, int harmnum, double fullrlo, double obszlo)
{
    /*********************************** 方式 0： *********************************************************/
    int BlkPerRow = (numrs_0*numzs_0-1+256)/256;
    Do_add_subharm_gpu_norz<<< BlkPerRow, 256 >>>(powers_out, fdata, numrs_0, numzs_0, fftlen, offset, norm, numharm, harmnum, fullrlo, obszlo);
}

static __global__ void Do_add_subharm_gpu_norz(float *powers_out, cufftComplex *fdata,
    int numrs_0, int numzs_0, int fftlen, int offset, float norm,
    int numharm, int harmnum, double fullrlo, double obszlo)
{
    const long long MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if (MYgtid >= numrs_0 * numzs_0) return;

    const int yy = MYgtid / numrs_0;      // z 方向索引 (0..numzs_0-1)
    const int xx = MYgtid - yy * numrs_0; // r 方向索引 (0..numrs_0-1)

    double harm_fract = (double)harmnum / (double)numharm;

    // ---- 计算 r 索引 addr_r ----
    double rr = fullrlo + xx * ACCEL_DR;                      // 基频 r 坐标
    double subr = calc_required_r_gpu(harm_fract, rr);       // 谐波数据中的浮点索引
    double drlo = calc_required_r_gpu(harm_fract, fullrlo);  // 起始点对应的浮点索引
    long long rlo = (long long)floor(drlo);                  // 基准整数索引
    unsigned short addr_r = index_from_r_gpu(subr, rlo);     // 最终绝对索引

    // ---- 计算 z 索引 addr_z ----
    double zz = obszlo + yy * ACCEL_DZ;                       // 基频 z 坐标
    double subz = calc_required_z_gpu(harm_fract, zz);       // 谐波数据中的浮点索引
    double dzlo = calc_required_z_gpu(harm_fract, obszlo);   // 起始点对应的浮点索引
    long long zlo = (long long)floor(dzlo);                  // 基准整数索引
    unsigned short addr_z = index_from_z_gpu(subz, zlo);     // 最终绝对索引

    // ---- 累加功率 ----
    Complex fdata_b = fdata[addr_z * fftlen + addr_r + offset];
    powers_out[MYgtid] += ((fdata_b.x * fdata_b.x + fdata_b.y * fdata_b.y) * norm);
}

void add_subharm_gpu_norz_1(float *powers_out, cufftComplex *fdata,
    int numrs_0, int numzs_0, int fftlen, int offset, float norm,
    double harm_fract, double drlo, long long rlo, double dzlo, long long zlo,
    double fullrlo, double obszlo)
{
    /*********************************** 方式 0： *********************************************************/
    int BlkPerRow = (numrs_0*numzs_0-1+256)/256;
    Do_add_subharm_gpu_norz_1<<< BlkPerRow, 256 >>>(powers_out, fdata, numrs_0, numzs_0, fftlen, offset, norm, harm_fract, drlo, rlo, dzlo, zlo, fullrlo, obszlo);
}

static __global__ void Do_add_subharm_gpu_norz_1(float *powers_out, cufftComplex *fdata,
    int numrs_0, int numzs_0, int fftlen, int offset, float norm,
    double harm_fract, double drlo, long long rlo, double dzlo, long long zlo,
    double fullrlo, double obszlo)
{
    const long long MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if (MYgtid >= numrs_0 * numzs_0) return;

    const int yy = MYgtid / numrs_0;      // z 方向索引
    const int xx = MYgtid - yy * numrs_0; // r 方向索引

    // r index
    double rr = fullrlo + xx * ACCEL_DR;
    double subr = rint(ACCEL_RDR * rr * harm_fract) * ACCEL_DR; //calc_required_r_gpu(harm_fract, rr);
    int addr_r = (int)((subr - rlo) * ACCEL_RDR + DBLCORRECT);

    // z index
    double zz = obszlo + yy * ACCEL_DZ;
    double subz = NEAREST_INT(ACCEL_RDZ * zz * harm_fract) * ACCEL_DZ; // calc_required_z_gpu(harm_fract, zz);
    int addr_z = (int)((subz - zlo) * ACCEL_RDZ + DBLCORRECT);

    // 累加功率
    Complex fdata_b = fdata[addr_z * fftlen + addr_r + offset];
    powers_out[MYgtid] += ((fdata_b.x * fdata_b.x + fdata_b.y * fdata_b.y) * norm);
}

// 双精度版本
static inline __device__ double calc_required_r_gpu(double harm_fract, double rfull)
/* Calculate the 'r' you need for subharmonic  */
/* harm_fract = harmnum / numharm if the       */
/* 'r' at the fundamental harmonic is 'rfull'. */
{
    return rint(ACCEL_RDR * rfull * harm_fract) * ACCEL_DR;
}

// 或者单精度版本（如果性能更重要）
static inline __device__ float calc_required_r_gpu_f(float harm_fract, float rfull)
{
    return rintf(ACCEL_RDR * rfull * harm_fract) * ACCEL_DR;
}


// 双精度版本
static inline __device__ int index_from_r_gpu(double r, double lor)
/* Return an index for a Fourier Freq given an array that */
/* has stepsize ACCEL_DR and low freq 'lor'.              */
{
    return (int) ((r - lor) * ACCEL_RDR + DBLCORRECT);
}

// 或者如果使用单精度浮点数
static inline __device__ int index_from_r_gpu_f(float r, float lor)
{
    return (int) ((r - lor) * ACCEL_RDR + DBLCORRECT);
}

// 双精度版本
static inline __device__ int calc_required_z_gpu(double harm_fract, double zfull)
/* Calculate the 'z' you need for subharmonic  */
/* harm_fract = harmnum / numharm if the       */
/* 'z' at the fundamental harmonic is 'zfull'. */
{
    return NEAREST_INT(ACCEL_RDZ * zfull * harm_fract) * ACCEL_DZ;
}

// 或者如果使用单精度浮点数
static inline __device__ int calc_required_z_gpu_f(float harm_fract, float zfull)
{
    return NEAREST_INT(ACCEL_RDZ * zfull * harm_fract) * ACCEL_DZ;
}


// 双精度版本
static inline __device__ int index_from_z_gpu(double z, double loz)
/* Return an index for a Fourier Fdot given an array that */
/* has stepsize ACCEL_DZ and low freq dot 'loz'.          */
{
    return (int) ((z - loz) * ACCEL_RDZ + DBLCORRECT);
}

// 单精度版本（如果性能更重要）
static inline __device__ int index_from_z_gpu_f(float z, float loz)
{
    return (int) ((z - loz) * ACCEL_RDZ + DBLCORRECT);
}

void add_subharm_gpu_end(float *powers_out, cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int numzs, int offset, float norm, accel_cand_gpu *cand_gpu_cpu, accel_cand_gpu * cand_array_search_gpu, int *nof_cand, float powcut, int numharm, int *cand_num_gpu)
{
    int BlkPerRow = (numrs_0*numzs_0-1+512)/512;
    Do_add_subharm_gpu_end<<< BlkPerRow, 512 >>>(powers_out, fdata, rinds, zinds, numrs_0, numzs_0, fftlen, offset, norm, cand_num_gpu, cand_array_search_gpu, powcut, numharm);

    // dim3 block(32,8);
    // dim3 grid((numrs_0+31)/32, (numzs_0+7)/8);
    // Do_add_subharm_gpu_end_fast<<<grid, block>>>(powers_out, fdata, rinds, zinds, numrs_0, numzs_0, fftlen, offset, norm, cand_num_gpu, cand_array_search_gpu, powcut, numharm);
}


static __global__ void Do_add_subharm_gpu_end(float *powers_out, cufftComplex *fdata,unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int offset, float norm, int *d_addr, accel_cand_gpu * cand_array_search_gpu, float powcut, int numharm)
{

    /*********************************** 方式 0 ： *********************************************************/
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numrs_0*numzs_0) return;

    const int yy = MYgtid/numrs_0 ;  // numzs_0_id
    unsigned short addr_z = __ldg(&zinds[yy]);
    const int xx = MYgtid -  yy * numrs_0 ; // numrs_0_id
    unsigned short addr_r = __ldg(&rinds[xx]);
    const int addr_result = addr_z * fftlen + addr_r + offset;


    Complex fdata_b = fdata[addr_result];
    // atomicAdd(&powers_out[MYgtid], ((fdata_b.x*fdata_b.x +  fdata_b.y*fdata_b.y)*norm));
    
    float total_power = (fdata_b.x*fdata_b.x +  fdata_b.y*fdata_b.y)*norm;
    powers_out[MYgtid] += total_power;
    total_power = powers_out[MYgtid];

    if(total_power > powcut)
    {
        int addr_search;
        addr_search = atomicAdd(&d_addr[0], 1);

        accel_cand_gpu cand_tmp ;
        cand_tmp.pow = total_power;
        cand_tmp.numharm = numharm;
        // cand_tmp.nof_cand = 1 ;
        cand_tmp.z_ind = yy;
        cand_tmp.r_ind = xx ;
        // cand_tmp.w_ind = 0;
        cand_array_search_gpu[ addr_search ] = cand_tmp ;
    }
}


static __global__ void Do_add_subharm_gpu_end_fast(
    float *__restrict__ powers_out,
    const cufftComplex *__restrict__ fdata,
    const unsigned short *__restrict__ rinds,
    const unsigned short *__restrict__ zinds,
    int numrs_0,
    int numzs_0,
    int fftlen,
    int offset,
    float norm,
    int *__restrict__ d_addr,
    accel_cand_gpu *__restrict__ cand_array_search_gpu,
    float powcut,
    int numharm)
{
     const int xx = blockIdx.x * blockDim.x + threadIdx.x;
    const int yy = blockIdx.y * blockDim.y + threadIdx.y;

    if (xx >= numrs_0 || yy >= numzs_0) return;

    const int MYgtid = yy * numrs_0 + xx;

    __shared__ unsigned short r_shared[32];
    __shared__ unsigned short z_shared[8];

    if (threadIdx.y == 0)
        r_shared[threadIdx.x] = rinds[xx];

    if (threadIdx.x == 0)
        z_shared[threadIdx.y] = zinds[yy];

    __syncthreads();

    unsigned short addr_r = r_shared[threadIdx.x];
    unsigned short addr_z = z_shared[threadIdx.y];

    cufftComplex fdata_b =
        fdata[addr_z * fftlen + addr_r + offset];

    float power =
        (fdata_b.x * fdata_b.x +
         fdata_b.y * fdata_b.y) * norm;

    powers_out[MYgtid] += power;
    power = powers_out[MYgtid];

    if(power > powcut)
    {
        int addr_search;
        addr_search = atomicAdd(&d_addr[0], 1);

        accel_cand_gpu cand_tmp ;
        cand_tmp.pow = power;
        cand_tmp.numharm = numharm;
        // cand_tmp.nof_cand = 1 ;
        cand_tmp.z_ind = yy;
        cand_tmp.r_ind = xx ;
        // cand_tmp.w_ind = 0;
        cand_array_search_gpu[ addr_search ] = cand_tmp ;
    }
}



void add_subharm_gpu_end_norz_1(
    float *powers_out,
    cufftComplex *fdata,
    int numrs_0,
    int numzs_0,
    int fftlen,
    int offset,
    float norm,
    accel_cand_gpu *cand_gpu_cpu,
    accel_cand_gpu *cand_array_search_gpu,
    int *nof_cand,
    float powcut,
    int numharm,
    int *cand_num_gpu,
    double harm_fract,
    double drlo,
    long long rlo,
    double dzlo,
    long long zlo,
    double fullrlo,
    double obszlo)
{
    int BlkPerRow = (numrs_0 * numzs_0 - 1 + 512) / 512;

    Do_add_subharm_gpu_end_norz_1<<<BlkPerRow, 512>>>(
        powers_out,
        fdata,
        numrs_0,
        numzs_0,
        fftlen,
        offset,
        norm,
        cand_num_gpu,
        cand_array_search_gpu,
        powcut,
        numharm,
        harm_fract,
        drlo,
        rlo,
        dzlo,
        zlo,
        fullrlo,
        obszlo);
}

static __global__ void Do_add_subharm_gpu_end_norz_1(
    float *powers_out,
    cufftComplex *fdata,
    int numrs_0,
    int numzs_0,
    int fftlen,
    int offset,
    float norm,
    int *d_addr,
    accel_cand_gpu *cand_array_search_gpu,
    float powcut,
    int numharm,
    double harm_fract,
    double drlo,
    long long rlo,
    double dzlo,
    long long zlo,
    double fullrlo,
    double obszlo)
{
    const long long MYgtid = blockDim.x * blockIdx.x + threadIdx.x;

    if (MYgtid >= (long long)numrs_0 * numzs_0) return;

    const int yy = MYgtid / numrs_0;
    const int xx = MYgtid - yy * numrs_0;

    // r index
    double rr = fullrlo + xx * ACCEL_DR;
    double subr = rint(ACCEL_RDR * rr * harm_fract) * ACCEL_DR;// calc_required_r_gpu(harm_fract, rr);

    int addr_r = (int)((subr - rlo) * ACCEL_RDR + DBLCORRECT);

    // z index
    double zz = obszlo + yy * ACCEL_DZ;
    double subz = NEAREST_INT(ACCEL_RDZ * zz * harm_fract) * ACCEL_DZ; // calc_required_z_gpu(harm_fract, zz);

    int addr_z = (int)((subz - zlo) * ACCEL_RDZ + DBLCORRECT);

    // read FFT data
    int addr_result = addr_z * fftlen + addr_r + offset;

    Complex fdata_b = fdata[addr_result];

    // power accumulation
    float total_power =
        (fdata_b.x * fdata_b.x + fdata_b.y * fdata_b.y) * norm;

    powers_out[MYgtid] += total_power;

    total_power = powers_out[MYgtid];

    // candidate detection
    if (total_power > powcut)
    {
        int addr_search = atomicAdd(&d_addr[0], 1);
        accel_cand_gpu cand_tmp;
        cand_tmp.pow = total_power;
        cand_tmp.numharm = numharm;
        cand_tmp.z_ind = yy;
        cand_tmp.r_ind = xx;

        cand_array_search_gpu[addr_search] = cand_tmp;
    }
}


void add_subharm_gpu_end_allharm(float *powers_out, cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, gpu_parameter_add *gpu_parameter_add_list, int numzs, accel_cand_gpu *cand_gpu_cpu, accel_cand_gpu * cand_array_search_gpu, int *nof_cand, float powcut, int numharm, int *cand_num_gpu, int fdata_len, int obs_corr_uselen, int obs_numz, int stage)
{
    int stage_base = (1 << (stage - 1)) - 1;
    int BlkPerRow = (numrs_0*numzs_0-1+256)/256;
    Do_add_subharm_gpu_end_allharm<<< BlkPerRow, 256 >>>(powers_out, fdata, rinds, zinds, numrs_0, numzs_0, gpu_parameter_add_list, cand_num_gpu, cand_array_search_gpu, powcut, numharm, fdata_len, obs_corr_uselen, obs_numz, stage_base);
}

static __global__ void Do_add_subharm_gpu_end_allharm(float *powers_out, cufftComplex *fdata,unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, gpu_parameter_add *gpu_parameter_add_list, int *d_addr, accel_cand_gpu * cand_array_search_gpu, float powcut, int numharm, int fdata_len, int obs_corr_uselen, int obs_numz, int stage)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numrs_0*numzs_0) return;

    const int yy = MYgtid/numrs_0 ;  // numzs_0_id
    const int xx = MYgtid -  yy * numrs_0 ; // numrs_0_id

    int ii;
    float total_power = 0.0f;

    #pragma unroll
    for (ii = 0; ii < numharm/2; ii++)
    {
        // const  gpu_parameter_add gpu_parameter_add_tmp = gpu_parameter_add_list[ii + numharm/2];
        const  gpu_parameter_add gpu_parameter_add_tmp = c_gpu_parameter_add[ii + numharm/2];

        const unsigned short addr_r = __ldg(&rinds[xx + (stage + ii)*obs_corr_uselen]);
        const unsigned short addr_z = __ldg(&zinds[yy + (stage + ii)*obs_numz]);
        const int addr_result = addr_z * gpu_parameter_add_tmp.fftlen + addr_r + gpu_parameter_add_tmp.offset;

        const  float2 fdata_b = __ldg(&fdata[addr_result + ii*fdata_len]);
        total_power += (fdata_b.x *fdata_b.x +  fdata_b.y*fdata_b.y)*gpu_parameter_add_tmp.norm_data;
    }
    // powers_out[MYgtid] += total_power;
    // total_power = powers_out[MYgtid];
    total_power += powers_out[MYgtid];
    powers_out[MYgtid] = total_power;

    // powers_out[MYgtid] = total_power;
    
    // if(total_power > powcut)
    // {
    //     int addr_search;
    //     addr_search = atomicAdd(&d_addr[0], 1);

    //     accel_cand_gpu cand_tmp ;
    //     cand_tmp.pow = total_power;
    //     cand_tmp.numharm = numharm;
    //     // cand_tmp.nof_cand = 1 ;
    //     cand_tmp.z_ind = yy;
    //     cand_tmp.r_ind = xx ;
    //     // cand_tmp.w_ind = 0;
    //     cand_array_search_gpu[ addr_search ] = cand_tmp ;
    // }
    {
        // 判断是否为候选
        bool is_cand = (total_power > powcut);
        accel_cand_gpu cand_tmp;
        if (is_cand) {
            cand_tmp.pow     = total_power;
            cand_tmp.numharm = numharm;
            cand_tmp.z_ind   = yy;
            cand_tmp.r_ind   = xx;
        }

        // ---- Warp 级聚合开始 ----
        // 1. 获取当前 warp 的候选掩码
        unsigned mask = __ballot_sync(0xffffffff, is_cand);
        
        // 2. 计算本线程在 warp 内的候选偏移量（紧凑排列）
        //    获取当前 lane 之前（不含自身）的候选数量
        unsigned lane_id = threadIdx.x % 32;                     // warp 内线程号
        unsigned mask_before = mask & ((1u << lane_id) - 1);     // 本线程之前的掩码
        unsigned warp_offset = __popc(mask_before);               // 本线程的块内序号
        
        // 3. 获取整个 warp 的候选总数
        unsigned warp_cand_count = __popc(mask);
        
        // 4. 由 lane 0 执行全局原子申请，并广播起始偏移
        int global_offset = 0;
        if (warp_cand_count > 0) {
            if (lane_id == 0) {
                global_offset = atomicAdd(&d_addr[0], warp_cand_count);
            }
            // 广播：将 lane 0 的 global_offset 广播给 warp 内所有线程
            global_offset = __shfl_sync(0xffffffff, global_offset, 0);
        }
        
        // 5. 候选线程写入全局数组
        if (is_cand) {
            cand_array_search_gpu[global_offset + warp_offset] = cand_tmp;
        }
        // ---- Warp 级聚合结束 ----
    }
}

void add_subharm_gpu_list(float *powers_out, cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int numzs, int offset, float norm, int readdatanum, int outpows_gpu_xlen, int obs_fftlen)
{
    int BlkPerRow = (numrs_0*numzs_0*readdatanum-1+256)/256;
    Do_add_subharm_gpu_list<<< BlkPerRow, 256 >>>(powers_out, fdata, rinds, zinds, numrs_0, numzs_0, fftlen, numzs, offset, norm, readdatanum, outpows_gpu_xlen, obs_fftlen);
    // CUDA_CHECK(cudaGetLastError());
}

static __global__ void Do_add_subharm_gpu_list(float *powers_out, cufftComplex *fdata,unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int numzs, int offset, float norm, int readdatanum, int outpows_gpu_xlen, int obs_fftlen)
{
    const long long  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numrs_0*numzs_0*readdatanum) return;

    const long long dataid = MYgtid/(numrs_0*numzs_0);
    const long long yy = (MYgtid-dataid*numrs_0*numzs_0)/numrs_0;
    unsigned short addr_z = __ldg(&zinds[yy]);
    int xx = MYgtid%numrs_0;
    unsigned short addr_r = __ldg(&rinds[xx]);

    
    const float2 fdata_b = __ldg(&fdata[dataid*fftlen*numzs +  addr_z * fftlen + addr_r + offset]);
    powers_out[dataid*outpows_gpu_xlen + yy*numrs_0 + xx] += ((fdata_b.x*fdata_b.x +  fdata_b.y*fdata_b.y)*norm); // power;
    // powers_out[dataid*outpows_gpu_xlen + yy*obs_fftlen + xx] += ((fdata_b.x*fdata_b.x +  fdata_b.y*fdata_b.y)*norm);
}


void add_subharm_gpu_cands_list(float *powers_out, cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int numzs, int offset, float norm, int readdatanum, int outpows_gpu_xlen, accel_cand_gpu *cand_gpu_cpu, accel_cand_gpu * cand_array_search_gpu, int *nof_cand, int output_x_max, float powcut, int numharm, int *cand_num_gpu, int obs_fftlen)
{

    int BlkPerRow = (numrs_0*numzs_0*readdatanum-1+512)/512;
    // dim3 dimGrid2D(BlkPerRow,readdatanum);
    Do_add_subharm_gpu_cands_list<<< BlkPerRow, 512 >>>(powers_out, fdata, rinds, zinds, numrs_0, numzs_0, fftlen, numzs, offset, norm, readdatanum, outpows_gpu_xlen, cand_num_gpu, cand_array_search_gpu, powcut, output_x_max, numharm, obs_fftlen);
}

static __global__ void Do_add_subharm_gpu_cands_list(float *powers_out, cufftComplex *fdata,unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int numzs, int offset, float norm, int readdatanum, int outpows_gpu_xlen, int *d_addr, accel_cand_gpu * cand_array_search_gpu, float powcut, int output_x_max, int numharm, int obs_fftlen)
{
    // const int dataid = blockIdx.y;
    const long long  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numrs_0*numzs_0*readdatanum) return;

    long long dataid = MYgtid/(numrs_0*numzs_0);
    long long yy = (MYgtid-dataid*numrs_0*numzs_0)/numrs_0;
    unsigned short addr_z = __ldg(&zinds[yy]);
    int xx = MYgtid%numrs_0;
    unsigned short addr_r = __ldg(&rinds[xx]);
    const long long addr_result = dataid*fftlen*numzs +  addr_z * fftlen + addr_r + offset;

    // float2 fdata_b = __ldg(reinterpret_cast<float2*>(&fdata[addr_result]));
    const  float2 fdata_b = __ldg(&fdata[addr_result]);

    // 使用__fmul_rn和__fadd_rn提高精度控制
    float real = fdata_b.x;
    float imag = fdata_b.y;
    float real_sq = __fmul_rn(real, real);
    float imag_sq = __fmul_rn(imag, imag);
    float power = __fmul_rn(__fadd_rn(real_sq, imag_sq), norm);

    float total_power = atomicAdd(&powers_out[dataid*outpows_gpu_xlen + yy*numrs_0 + xx], power);
    // float total_power = atomicAdd(&powers_out[dataid*outpows_gpu_xlen + yy*obs_fftlen + xx], power);
    total_power += power;

    if(total_power > powcut)
    {
        int addr_search;
        addr_search = atomicAdd(&d_addr[dataid], 1);

        accel_cand_gpu cand_tmp ;
        cand_tmp.pow = total_power;
        cand_tmp.numharm = numharm;
        // cand_tmp.nof_cand = 1;
        cand_tmp.z_ind = yy;
        cand_tmp.r_ind = xx;
        // cand_tmp.w_ind = 0;
        cand_array_search_gpu[ addr_search + dataid*output_x_max] = cand_tmp ;
    }
    // else return;
}


void add_subharm_gpu_cands_list_allharm(float *powers_out, cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, gpu_parameter_add *gpu_parameter_add_list, int numzs, int readdatanum, int outpows_gpu_xlen, accel_cand_gpu *cand_gpu_cpu, accel_cand_gpu * cand_array_search_gpu, int *nof_cand, int output_x_max, float powcut, int numharm, int *cand_num_gpu, int obs_fftlen, int fdata_len, int obs_corr_uselen, int obs_numz, int stage)
{

    int BlkPerRow = (numrs_0*numzs_0*readdatanum-1+256)/256;
    // dim3 dimGrid2D(BlkPerRow,readdatanum);
    Do_add_subharm_gpu_cands_list_allharm<<< BlkPerRow, 256 >>>(powers_out, fdata, rinds, zinds, numrs_0, numzs_0, gpu_parameter_add_list, numzs, readdatanum, outpows_gpu_xlen, cand_num_gpu, cand_array_search_gpu, powcut, output_x_max, numharm, obs_fftlen, fdata_len, obs_corr_uselen, obs_numz, (pow(2,stage-1)-1));
}

static __global__ void Do_add_subharm_gpu_cands_list_allharm(float *powers_out, cufftComplex *fdata,unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, gpu_parameter_add *gpu_parameter_add_list, int numzs, int readdatanum, int outpows_gpu_xlen, int *d_addr, accel_cand_gpu * cand_array_search_gpu, float powcut, int output_x_max, int numharm, int obs_fftlen, int fdata_len, int obs_corr_uselen, int obs_numz, int stage)
{
    // const int dataid = blockIdx.y;
    const long long  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numrs_0*numzs_0*readdatanum) return;

    long long dataid = MYgtid/(numrs_0*numzs_0);
    long long yy = (MYgtid-dataid*numrs_0*numzs_0)/numrs_0;
    int xx = MYgtid%numrs_0;

    int ii;
    long long addr_result_0 = dataid*outpows_gpu_xlen + yy*numrs_0 + xx;
    float total_power = 0.0f;

    #pragma unroll
    for(ii=0; ii<numharm/2; ii++)
    { 
        const gpu_parameter_add gpu_parameter_add_tmp = c_gpu_parameter_add[ii + numharm/2];
        const  unsigned short addr_r = __ldg(&rinds[xx + (stage + ii)*obs_corr_uselen]);
        const  unsigned short addr_z = __ldg(&zinds[yy + (stage + ii)*obs_numz]);

        // const long long addr_result = dataid * gpu_parameter_add_tmp.fftlen * numzs + addr_z * gpu_parameter_add_tmp.fftlen + addr_r + gpu_parameter_add_tmp.offset;

	const long long addr_result = dataid * gpu_parameter_add_tmp.fftlen * gpu_parameter_add_tmp.numkern + addr_z * gpu_parameter_add_tmp.fftlen + addr_r + gpu_parameter_add_tmp.offset;

        const float2 fdata_b = __ldg(&fdata[addr_result + ii*fdata_len]);
        total_power += (fdata_b.x *fdata_b.x +  fdata_b.y*fdata_b.y)*gpu_parameter_add_tmp.norm_data;
    }
    powers_out[addr_result_0] += total_power;
    total_power = powers_out[addr_result_0];

    // powers_out[addr_result_0] = total_power;

    if(total_power > powcut)
    {
        int addr_search;
        addr_search = atomicAdd(&d_addr[dataid], 1);

        accel_cand_gpu cand_tmp ;
        cand_tmp.pow = total_power;
        cand_tmp.numharm = numharm;
        // cand_tmp.nof_cand = 1;
        cand_tmp.z_ind = yy;
        cand_tmp.r_ind = xx;
        // cand_tmp.w_ind = 0;
        cand_array_search_gpu[ addr_search + dataid*output_x_max] = cand_tmp ;
    }
}


void inmem_add_ffdotpows_gpu_gpu(float *fdp, float *powptr, int *rinds, int zlo, int numrs, int numzs, int stage, long long rlen)
{
    int BlkPerRow = (numrs*numzs-1+512)/512;
    Do_inmem_add_ffdotpows_gpu_gpu<<< BlkPerRow, 512>>>(fdp, powptr, rinds, zlo, numrs, numzs, (int)(pow(2,stage-1)), rlen);
}

static  __global__ void Do_inmem_add_ffdotpows_gpu_gpu(float *fdp, float *powptr, int *rinds,int zlo, int numrs, int numzs, int stage, long long rlen)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numrs*numzs) return;

    const int ii = MYgtid/numrs;
    const int jj = MYgtid - ii*numrs;
    int zz = zlo + ii * 2;
    int zind, subz;
    int kk;
    float tmp = 0.0f;
    
    for(kk=0; kk<stage; kk++)
    {
        subz = calc_required_z_gpu((2.0*kk+1.0)/(stage*2), zz);
        zind = index_from_z_gpu(subz, zlo);
        int offset = zind * rlen;
        tmp += fdp[offset+rinds[jj+ (kk+stage-1)*numrs]];
    }
    powptr[MYgtid] += tmp;
}

void inmem_add_ffdotpows_gpu_gpu_list(float *fdp, float *powptr, int *rinds, int zlo, int numrs, int numzs, int stage, long long rlen, int outpows_gpu_xlen, int readdatanum, int outpows_gpu_obs_xlen)
{
    int BlkPerRow = (numrs*numzs*readdatanum-1+512)/512;
    Do_inmem_add_ffdotpows_gpu_gpu_list<<< BlkPerRow, 512>>>(fdp, powptr, rinds, zlo, numrs, numzs, (int)(pow(2,stage-1)), rlen, outpows_gpu_xlen, readdatanum, outpows_gpu_obs_xlen);
}

static  __global__ void Do_inmem_add_ffdotpows_gpu_gpu_list(float *fdp, float *powptr, int *rinds,int zlo, int numrs, int numzs, int stage, long long rlen, int outpows_gpu_xlen, int readdatanum, int outpows_gpu_obs_xlen)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numrs*numzs*readdatanum) return;

    const int dataid = MYgtid/(numrs*numzs);
    const int ii = (MYgtid - dataid*numrs*numzs)/numrs;
    const int jj = MYgtid%numrs;

    int zz = zlo + ii * 2;
    int zind, subz;
    int kk;
    float tmp = 0.0f;
    
    for(kk=0; kk<stage; kk++)
    {
        subz = calc_required_z_gpu((2.0*kk+1.0)/(stage*2), zz);
        zind = index_from_z_gpu(subz, zlo);
        int offset = zind * rlen;
        tmp += fdp[offset+rinds[jj+ (kk+stage-1)*numrs] + dataid*outpows_gpu_obs_xlen];
    }
    // powptr[MYgtid] += tmp;
    powptr[dataid*outpows_gpu_xlen + ii*numrs + jj] += tmp;
}

void inmem_add_subharm_and_search_ffdotpows_gpu_list_m(float *outpows_gpu_obs, float *outpows_gpu, int *rinds_gpu, int fd_zlo, int fd_numrs, int fd_numzs, int stage, long long obs_rlen, accel_cand_gpu * cand_array_search_gpu, accel_cand_gpu *cand_gpu_cpu, float powcut, int output_x_max, int *nof_cand, long long outpows_gpu_obs_xlen, int readdatanum, int *cand_num_gpu, int harmtosum)
{
    int BlkPerRow = (fd_numrs*fd_numzs*readdatanum-1+512)/512;
    Do_inmem_add_subharm_and_search_ffdotpows_gpu_list_m<<<BlkPerRow, 512>>>(outpows_gpu_obs, outpows_gpu, rinds_gpu, fd_zlo, fd_numrs, fd_numzs, (int)(pow(2,stage-1)), obs_rlen, powcut, cand_array_search_gpu, output_x_max, outpows_gpu_obs_xlen, readdatanum, cand_num_gpu, harmtosum);
}

static  __global__ void Do_inmem_add_subharm_and_search_ffdotpows_gpu_list_m(float *outpows_gpu_obs, float *outpows_gpu, int *rinds,int zlo, int numrs, int numzs, int stage, long long rlen, float powcut, accel_cand_gpu * cand_array_search_gpu, int output_x_max, long long outpows_gpu_obs_xlen, int readdatanum, int *cand_num_gpu, int harmtosum)
{
    const int  MYgtid_0 = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid_0>=numrs*numzs*readdatanum) return;

    int dataid = MYgtid_0/(numrs*numzs);
    const int  MYgtid = MYgtid_0 - dataid*numrs*numzs;
    if(MYgtid>=numrs*numzs) return;

    const int ii = MYgtid/numrs;
    const int jj = MYgtid - ii*numrs;
    int zz = zlo + ii * 2;
    int zind, subz;
    int kk;
    float tmp = 0.0f;
    
    for(kk=0; kk<stage; kk++)
    {
        subz = calc_required_z_gpu((2.0*kk+1.0)/(stage*2), zz);
        zind = index_from_z_gpu(subz, zlo);
        const long long offset = zind * rlen;
        tmp += outpows_gpu_obs[outpows_gpu_obs_xlen*dataid + offset+rinds[jj+ (kk+stage-1)*numrs]];
    }
    outpows_gpu[MYgtid + dataid*output_x_max] += tmp;
    tmp = outpows_gpu[MYgtid + dataid*output_x_max];


    if(tmp > powcut)
    {
        int addr_search;

        addr_search = atomicAdd(&cand_num_gpu[dataid], 1);

        accel_cand_gpu cand_tmp ;
        cand_tmp.pow = tmp ;
        cand_tmp.numharm = harmtosum;
        // cand_tmp.nof_cand = 1 ;
        cand_tmp.z_ind = ii;
        cand_tmp.r_ind = jj;
        // cand_tmp.w_ind = 0;
        cand_array_search_gpu[ addr_search + dataid*output_x_max] = cand_tmp ;
    }
}


int  search_ffdotpows_gpu(float powcut, float *d_fundamental, accel_cand_gpu * cand_array_search_gpu, int numzs, int numrs, accel_cand_gpu *cand_gpu_cpu)
{
	static int  *d_addr;
    static int firsttime = 1;
	int h_addr;
	
    if(firsttime)
    {
        cudaMalloc((void **)&d_addr, sizeof(int) * 1);
        firsttime = 0;
    }
	cudaMemset(d_addr, 0, sizeof(int)); // set d_addr to 0

	int BlkPerRow=(numrs*numzs-1+512)/512;
	search_ffdotpows_kernel<<<BlkPerRow, 512>>>(powcut, d_fundamental, cand_array_search_gpu, numzs, numrs, d_addr);
    // CUDA_CHECK(cudaGetLastError());
	cudaMemcpy(&h_addr, d_addr, sizeof(int) * 1, cudaMemcpyDeviceToHost);	
	cudaMemcpy(cand_gpu_cpu, cand_array_search_gpu, sizeof(accel_cand_gpu) * h_addr, cudaMemcpyDeviceToHost);
    
	return h_addr ;
}

static __global__ void  search_ffdotpows_kernel(float powcut, float *d_fundamental, accel_cand_gpu * cand_array_search_gpu, int numzs, int numrs, int *d_addr)
{
    const int MYgtid = blockIdx.x * blockDim.x + threadIdx.x;
    if(MYgtid >= numrs*numzs) return;
    
    float pow = d_fundamental[MYgtid];
    if(pow > powcut)
    {
        int addr_search;
        int z_ind;

        addr_search = atomicAdd(&d_addr[0], 1);

        accel_cand_gpu cand_tmp ;
        cand_tmp.pow = pow ;
        // cand_tmp.nof_cand = 1 ;
        z_ind = (int)(MYgtid/numrs);
        cand_tmp.z_ind = z_ind;
        cand_tmp.r_ind = MYgtid - z_ind * numrs ;
        // cand_tmp.w_ind = 0;
        cand_array_search_gpu[ addr_search ] = cand_tmp ;
    }
}

void inmem_add_subharm_and_search_ffdotpows_gpu(float *outpows_gpu_obs, float *outpows_gpu, int *rinds_gpu, int fd_zlo, int fd_numrs, int fd_numzs, int stage, long long obs_rlen, accel_cand_gpu * cand_array_search_gpu, accel_cand_gpu *cand_gpu_cpu, float powcut, int numharm, int *cand_num_gpu)
{

    // static int  *d_addr;
    // static int firsttime = 1;
    // int h_addr;

    // if(firsttime)
    // {
    //     cudaMalloc((void **)&d_addr, sizeof(int) * 1);
    //     firsttime = 0;
    // }
	// cudaMemset(d_addr, 0, sizeof(int)); // set d_addr to 0

    int BlkPerRow=(fd_numzs*fd_numrs-1+512)/512;
    Do_inmem_add_subharm_and_search_ffdotpows_gpu<<<BlkPerRow, 512>>>(outpows_gpu_obs, outpows_gpu, rinds_gpu, fd_zlo, fd_numrs, fd_numzs, (int)(pow(2,stage-1)), obs_rlen, powcut, cand_num_gpu, cand_array_search_gpu, numharm);


    // cudaMemcpy(&h_addr, d_addr, sizeof(int) * 1, cudaMemcpyDeviceToHost);	

    // if(h_addr)
	//     cudaMemcpy(cand_gpu_cpu, cand_array_search_gpu, sizeof(accel_cand_gpu) * h_addr, cudaMemcpyDeviceToHost);
    
	// return h_addr ;
}


static  __global__ void Do_inmem_add_subharm_and_search_ffdotpows_gpu(float *outpows_gpu_obs, float *outpows_gpu, int *rinds,int zlo, int numrs, int numzs, int stage, long long rlen, float powcut, int *d_addr, accel_cand_gpu * cand_array_search_gpu, int numharm)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numrs*numzs) return;

    const int ii = MYgtid/numrs;
    const int jj = MYgtid - ii*numrs;
    int zz = zlo + ii * 2;
    int zind, subz;
    int kk;
    float tmp = 0.0f;
    
    for(kk=0; kk<stage; kk++)
    {
        subz = calc_required_z_gpu((2.0*kk+1.0)/(stage*2), zz);
        zind = index_from_z_gpu(subz, zlo);
        int offset = zind * rlen;
        tmp += outpows_gpu_obs[offset+rinds[jj+ (kk+stage-1)*numrs]];
    }
    outpows_gpu[MYgtid] += tmp;
    tmp = outpows_gpu[MYgtid];


    if(tmp > powcut)
    {
        int addr_search;
        int z_ind;

        addr_search = atomicAdd(&d_addr[0], 1);

        accel_cand_gpu cand_tmp ;
        cand_tmp.pow = tmp ;
        cand_tmp.numharm = numharm;
        // cand_tmp.nof_cand = 1 ;
        z_ind = (int)(MYgtid/numrs);
        cand_tmp.z_ind = z_ind;
        cand_tmp.r_ind = MYgtid - z_ind * numrs ;
        // cand_tmp.w_ind = 0;
        cand_array_search_gpu[ addr_search ] = cand_tmp ;
    }
}



void  search_ffdotpows_gpu_list_m(float powcut, float *d_fundamental, accel_cand_gpu * cand_array_search_gpu, int numzs, int numrs, accel_cand_gpu *cand_gpu_cpu, int readdatanum, int *nof_cand, int output_x_max, int d_fundamental_xlen)
{
    static int  *d_addr;
    static int firsttime = 1;
	
    if(firsttime)
    {
        cudaMalloc((void **)&d_addr, sizeof(int) * readdatanum);
        firsttime = 0;
    }

	cudaMemset(d_addr, 0, sizeof(int)*readdatanum); // set d_addr to 0
    
	int BlkPerRow=(numrs*numzs*readdatanum-1+512)/512;
	search_ffdotpows_kernel_list_m<<<BlkPerRow, 512>>>(powcut, d_fundamental, cand_array_search_gpu, numzs, numrs, d_addr, readdatanum, output_x_max, d_fundamental_xlen);
    
	cudaMemcpy(nof_cand, d_addr, sizeof(int) * readdatanum, cudaMemcpyDeviceToHost);	
    
    
    // cudaMemcpy(cand_gpu_cpu, cand_array_search_gpu, sizeof(accel_cand_gpu)* output_x_max*readdatanum, cudaMemcpyDeviceToHost);

    int ii;
    for(ii=0; ii<readdatanum; ii++)
    {
        cudaMemcpy(cand_gpu_cpu+ii*output_x_max, cand_array_search_gpu+ii*output_x_max, sizeof(accel_cand_gpu)* nof_cand[ii], cudaMemcpyDeviceToHost);
    }
}


static __global__ void  search_ffdotpows_kernel_list_m(float powcut, float *d_fundamental, accel_cand_gpu * cand_array_search_gpu, int numzs, int numrs, int *d_addr, int readdatanum, int put_x_max, int d_fundamental_xlen)
{
    const int MYgtid = blockIdx.x * blockDim.x + threadIdx.x;
    if(MYgtid >= numrs*numzs*readdatanum) return;
    
    int d_f_id = MYgtid / (numrs*numzs);
    int yy = (MYgtid-d_f_id*numrs*numzs)/numrs;
    int xx = MYgtid%numrs;

    float pow = __ldg(&d_fundamental[d_f_id*d_fundamental_xlen + yy*numrs + xx]);

    
    if(pow > powcut)
    {
        int addr_search;

        addr_search = atomicAdd(&d_addr[d_f_id], 1);

        accel_cand_gpu cand_tmp ;
        cand_tmp.pow = pow ;
        // cand_tmp.numharm = numharm;
        // cand_tmp.nof_cand = 1;
        cand_tmp.z_ind = yy;
        cand_tmp.r_ind = xx;
        // cand_tmp.w_ind = 0;
        cand_array_search_gpu[ addr_search + d_f_id*put_x_max] = cand_tmp ;
    }
}





void hunt_GPU(double *xx, int n, double x, int *jlo)
{
    int jm, jhi, inc;
    int ascnd;

    ascnd = (xx[n] >= xx[1]);
    if (*jlo <= 0 || *jlo > n) {
        *jlo = 0;
        jhi = n + 1;
    } else {
        inc = 1;
        if ((x >= xx[*jlo]) == ascnd) {
            if (*jlo == n)
                return;
            jhi = (*jlo) + 1;
            while ((x >= xx[jhi]) == ascnd) {
                *jlo = jhi;
                inc += inc;
                jhi = (*jlo) + inc;
                if (jhi > n) {
                    jhi = n + 1;
                    break;
                }
            }
        } else {
            if (*jlo == 1) {
                *jlo = 0;
                return;
            }
            jhi = (*jlo)--;
            while ((x < xx[*jlo]) == ascnd) {
                jhi = (*jlo);
                inc <<= 1;
                if (inc >= jhi) {
                    *jlo = 0;
                    break;
                } else
                    *jlo = jhi - inc;
            }
        }
    }
    while (jhi - (*jlo) != 1) {
        jm = (jhi + (*jlo)) >> 1;
        if ((x >= xx[jm]) == ascnd)
            *jlo = jm;
        else
            jhi = jm;
    }
    if (x == xx[n])
        *jlo = n - 1;
    if (x == xx[1])
        *jlo = 1;
}

void hunt_CPU(double *xx, int n, double x, int *jlo)
{
    int jm, jhi, inc;
    int ascnd;

    ascnd = (xx[n] >= xx[1]);
    if (*jlo <= 0 || *jlo > n) {
        *jlo = 0;
        jhi = n + 1;
    } else {
        inc = 1;
        if ((x >= xx[*jlo]) == ascnd) {
            if (*jlo == n)
                return;
            jhi = (*jlo) + 1;
            while ((x >= xx[jhi]) == ascnd) {
                *jlo = jhi;
                inc += inc;
                jhi = (*jlo) + inc;
                if (jhi > n) {
                    jhi = n + 1;
                    break;
                }
            }
        } else {
            if (*jlo == 1) {
                *jlo = 0;
                return;
            }
            jhi = (*jlo)--;
            while ((x < xx[*jlo]) == ascnd) {
                jhi = (*jlo);
                inc <<= 1;
                if (inc >= jhi) {
                    *jlo = 0;
                    break;
                } else
                    *jlo = jhi - inc;
            }
        }
    }
    while (jhi - (*jlo) != 1) {
        jm = (jhi + (*jlo)) >> 1;
        if ((x >= xx[jm]) == ascnd)
            *jlo = jm;
        else
            jhi = jm;
    }
    if (x == xx[n])
        *jlo = n - 1;
    if (x == xx[1])
        *jlo = 1;
}




// __global__ static void add_to_prof(double *prof_gpu, double *buffer_gpu, int N,
//     long double *lophase_gpu, long double *deltaphase_gpu,
//     double *dataval, double *phaseadded_gpu, int onbin, int offbin, int nsub, int worklen)
// // This routine adds a data sample of size dataval and phase
// // duration deltaphase to the buffer (and possibly the profile
// // prof) of length N.  The starting phase is lophase, and the
// // running total of the amount added to the buffer is in
// // *phaseadded (0-1).  It should be set to 0.0 the first time
// // this routine is called.
// //
// // This routine uses the "standard" PRESTO method of "drizzling"
// // finite duration samples into as many profile bins as it would
// // cover in time.  This leads to some amount of correlation
// // between the profile bins.  If you don't want that, try using
// // add_to_prof_sample() instead, which assumes delta function samples.
// {
//     const int MYgtid = blockIdx.x * blockDim.x + threadIdx.x;
//     if(MYgtid >=  nsub*(offbin-onbin+1)) return;

//     int currentsub = MYgtid/(offbin-onbin+1);
//     int currentbin = MYgtid%(offbin-onbin+1);

//     int ii, icurbin;
//     double curbin, dphs, fdphs, onempadd;
//     const double profbinwidth = 1.0 / N;
//     double deltaphase = deltaphase_gpu[currentbin];
//     const double valperphs = dataval[currentbin+currentsub*(offbin-onbin+1)] / deltaphase;

//     double phaseadded = phaseadded_gpu[currentsub];
//     double *buffer = buffer_gpu+currentbin+currentsub*(offbin-onbin+1);
//     double *prof = prof_gpu+currentbin+currentsub*(offbin-onbin+1);


//     // Note:  The buffer is always synced in phase with prof.  We 
//     //   dump and clear it when 1 full wrap of data has been included.
//     curbin = lophase_gpu[currentbin] * N; // double

//     while (deltaphase > 1e-12) { // Close to or above zero
//         // The integer bin number
//         icurbin = (int) floor(curbin + 1e-12);
//         // Amount of phase we can get in current bin
//         dphs = ((icurbin + 1.0) - curbin) * profbinwidth;
//         // Make sure that icurbin is not outside bounds (can happen
//         // because of the floor(curbin + 1e-12) line above)
//         icurbin %= N;
//         // All of the sample can go in the current bin
//         if (dphs > deltaphase) dphs = deltaphase;
//         // How much phase is left to fill the buffer
//         onempadd = (1.0 - phaseadded);
//         // Will we need to dump the buffer?
//         fdphs = onempadd > dphs ? dphs : onempadd;
//         buffer[icurbin] += fdphs * valperphs;
//         phaseadded += fdphs;
//         // For debugging....
//         //printf("%4d  %10.6g  %10.6g  %10.6g  %10.6g  %10.6g\n",
//         //    icurbin, curbin, deltaphase, fdphs, fdphs * valperphs, *phaseadded);
//         if (fabs((phaseadded) - 1.0) < 1e-12) { // Need to dump buffer
//             // Dump the buffer into the profile array
//             for (ii = 0; ii < N; ii++) prof[ii] += buffer[ii];
//             // Reset the buffer array to zeros
//             for (ii = 0; ii < N; ii++) buffer[ii] = 0.0;
//             // Now add the rest of the frac to the new buffer
//             fdphs = dphs - onempadd;
//             buffer[icurbin] += fdphs * valperphs;
//             // And correct the phase added
//             phaseadded = fdphs;
//             // For debugging....
//             //printf("----  %10.6g  %10.6g  %10.6g  %10.6g  %10.6g\n",
//             //    curbin, deltaphase, fdphs, fdphs * valperphs, *phaseadded);
//         }
//         deltaphase -= dphs;
//         curbin += dphs * N;
//     }
// }


__global__ static void add_to_prof_sample(double *prof, double *buffer, int N,
long double *lophase, long double *deltaphase, float *dataval, int onbin, int offbin, int nsub, int worklen)
// This routine adds a data sample of size dataval and phase
// duration deltaphase to the buffer (and possibly the profile
// prof) of length N.  The starting phase is lophase.
//
// This routine assumes that the sample is a delta function
// and so will put it at phase lophase + 0.5 * deltaphase.
// The buffer is used to keep track of how many samples have been
// placed in each bin.  This is how most other codes fold data/
{
    // The integer bin number.  The mod is necessary due to  "+ 1e-12"

    const int MYgtid = blockIdx.x * blockDim.x + threadIdx.x;
    if(MYgtid >=  nsub*(offbin-onbin+1)) return;
    int currentsub = MYgtid/(offbin-onbin+1);
    int currentbin = MYgtid%(offbin-onbin+1);


    // int bin = ((int) floor((lophase[currentbin] + 0.5 * deltaphase[currentbin]) * N + 1e-12)) % N;

    long double phase_bk = ((lophase[currentbin] + 0.5 * deltaphase[currentbin]) * N + 1e-12);

    int binLarger0 = (phase_bk > 0.0L) ? 1:0;
    int bin;

    if(binLarger0)
        bin = ((int)phase_bk) %N;
    else
        bin = ((int)phase_bk-1) %N;

    atomicAdd(&buffer[bin], 1.0);
    atomicAdd(&prof[bin], dataval[currentbin + onbin + currentsub*worklen]);

    // buffer[bin] += 1.0;
    // prof[bin] += dataval[currentbin + onbin + currentsub*worklen];
    // return;
}


double fold_gpu_cu(float *data, int nsub, int numdata, double dt, double tlo,
            double *prof, int numprof, double startphs,
            double *buffer, double *phaseadded,
            double fo, double fdot, double fdotdot, int flags,
            double *delays, double *delaytimes, int numdelays,
            int *onoffpairs, foldstats_gpu * stats, int standard, int ONOFF, int DELAYS, int worklen)
/* This routine is a general pulsar folding algorithm.  It will fold  */
/* data for a pulsar with single and double frequency derivatives and */
/* with arbitrary pulse delays (for example: variable time delays     */
/* due to light travel time in a binary).  These delays are described */
/* in the arrays '*delays' and '*delaytimes'. The folding may also be */
/* turned on and off throughout the data by using 'onoffpairs'. The   */
/* profile will have the data corresponding to time 'tlo' placed at   */
/* the phase corresponding to time 'tlo' using 'fo', 'fdot', and      */
/* 'fdotdot' plus 'startphs' and the appropriate delay.               */
/* Arguments:                                                         */
/*    'data' is a float array containing the data to fold.            */
/*    'numdata' is the number of points in *data.                     */
/*    'dt' is the time duration of each data bin.                     */
/*    'tlo' is the time of the start of the 1st data pt.              */
/*    'prof' is a double prec array to contain the profile.           */
/*    'numprof' is the length of the profile array.                   */
/*    'startphs'is the phase offset [0-1] for the first point.        */
/*    'buffer' is a double prec array of numprof values containing    */
/*            data that hasn't made it into the prof yet.             */
/*    'phaseadded' is the address to a variable showing how much      */
/*            has been added to the buffer [0-1] (must start as 0.0)  */
/*    'fo' the starting frequency to fold.                            */
/*    'fdot' the starting frequency derivative.                       */
/*    'fdotdot' the frequency second derivative.                      */
/*    'flags' is an integer containing flags of how to fold:          */
/*            0 = No *delays and no *onoffpairs                       */
/*            1 = Use *delays but no *onoffpairs                      */
/*            2 = No *delays but use *onoffpairs                      */
/*            3 = Use *delays and use *onoffpairs                     */
/*    'delays' is an array of time delays.                            */
/*    'delaytimes' are the times where 'delays' were calculated.      */
/*    'numdelays' is how many points are in 'delays' and 'delaytimes' */
/*    'onoffpairs' is array containing pairs of numbers that          */
/*            represent the bins when we will actively add            */
/*            to the profile.  To fold the whole array,               */
/*            onoffpairs should be [0, numdata-1].                    */
/*    'stats' are statistics of the data that were folded as well     */
/*            as the folded profile itself.  If this                  */
/*            routine is used on consecutive pieces of the            */
/*            same data, fold() will use the current values           */
/*            and update them at the end of each call.                */
/*            So each parameter must be set to 0.0 before             */
/*            fold() is called for the first time.                    */
/*    'standard' If true, uses classic prepfold 'drizzling'           */
/*            Otherwise, adds full sample to nearest bin.             */
/* Notes:  fo, fdot, and fdotdot correspond to 'tlo' = 0.0            */
/*    (i.e. to the beginning of the first data point)                 */
{
    int ii, jj, onbin, offbin, *onoffptr = NULL;
    int arrayoffset = 0;
    long double phase, phasenext = 0.0, deltaphase, T, Tnext, TD, TDnext;
    long double profbinwidth, lophase, hiphase;
    double dev, delaytlo = 0.0, delaythi = 0.0, delaylo = 0.0, delayhi = 0.0;
    double *delayptr = NULL, *delaytimeptr = NULL, dtmp;
    float data_bk;

    long double *lophase_gpu, *deltaphase_gpu;
    static int firsttime=1;
    int onoffbin;



    /* Initialize some variables and save some FLOPs later... */

    fdot /= 2.0;
    fdotdot /= 6.0;
    profbinwidth = 1.0 / numprof;
    if (ONOFF)
        onoffptr = onoffpairs;
    // stats->numprof = (double) numprof;
    // stats->data_var *= (stats->numdata - 1.0);


    do {                        /* Loop over the on-off pairs */
        /* Set the on-off pointers and variables */
        if (ONOFF) {
            onbin = *onoffptr;
            offbin = *(onoffptr + 1);
            onoffptr += 2;
        } else {
            onbin = 0;
            offbin = numdata - 1;
        }

        if(firsttime)
        {
            cudaMalloc((void**)&lophase_gpu, sizeof(double)*(offbin-onbin+1));
            cudaMalloc((void**)&deltaphase_gpu, sizeof(double)*(offbin-onbin+1));
            firsttime=0;
            onoffbin = offbin-onbin+1;
        }
        else if(onoffbin != offbin-onbin+1)
        {
            cudaFree(lophase_gpu);
            cudaFree(deltaphase_gpu);
            cudaMalloc((void**)&lophase_gpu, sizeof(double)*(offbin-onbin+1));
            cudaMalloc((void**)&deltaphase_gpu, sizeof(double)*(offbin-onbin+1));
            onoffbin = offbin-onbin+1;
        }

        /* Initiate the folding start time */
        T = tlo + onbin * dt;
        TD = T;

        /* Set the delay pointers and variables */
        if (DELAYS) {
            /* Guess that the next delay we want is the next available */
            arrayoffset += 2;   /* Beware nasty NR zero-offset kludges! */
            hunt_CPU(delaytimes - 1, numdelays, T, &arrayoffset);
            arrayoffset--;
            delaytimeptr = delaytimes + arrayoffset;
            delayptr = delays + arrayoffset;
            delaytlo = *delaytimeptr;
            delaythi = *(delaytimeptr + 1);
            delaylo = *delayptr;
            delayhi = *(delayptr + 1);

            /* Adjust the folding start time for the delays */
            TD -= LININTERP(TD, delaytlo, delaythi, delaylo, delayhi);
        }

        /* Get the starting pulsar phase (cyclic). */
        phase = TD * (TD * (TD * fdotdot + fdot) + fo) + startphs;
        lophase = (phase < 0.0) ? 1.0 + modf(phase, &dtmp) : modf(phase, &dtmp);

        /* Generate the profile for this onoff pair */
        for (ii = onbin; ii <= offbin; ii++)
        {
            /* Calculate the barycentric time for the next point. */
            Tnext = tlo + (ii + 1) * dt;
            TDnext = Tnext;
            
            // /* Set the delay pointers and variables */
            if (DELAYS) {
                if (Tnext > delaythi) {
                    /* Guess that the next delay we want is the next available */
                    arrayoffset += 2;   /* Beware nasty NR zero-offset kludges! */
                    hunt_GPU(delaytimes - 1, numdelays, Tnext, &arrayoffset);
                    arrayoffset--;
                    delaytimeptr = delaytimes + arrayoffset;
                    delayptr = delays + arrayoffset;
                    delaytlo = *delaytimeptr;
                    delaythi = *(delaytimeptr + 1);
                    delaylo = *delayptr;
                    delayhi = *(delayptr + 1);
                }
                /* Adjust the folding start time for the delays */
                TDnext -= LININTERP(Tnext, delaytlo, delaythi, delaylo, delayhi);
            }
            /* Get the pulsar phase (cyclic) for the next point. */
            phasenext = TDnext * (TDnext * (TDnext * fdotdot + fdot) + fo) + startphs;
            /* How much total phase does the data point cover? */
            deltaphase = phasenext - phase;

            // if(ii==0)
            // printf("\n%d    %d    %d    %Le    %Le    %Le    %Le",ii, DELAYS, numprof, lophase, phase, deltaphase, 1.0L*numprof*lophase);

            cudaMemcpy(&lophase_gpu[ii-onbin], &lophase, sizeof(long double), cudaMemcpyHostToDevice);  
            cudaMemcpy(&deltaphase_gpu[ii-onbin], &deltaphase, sizeof(long double), cudaMemcpyHostToDevice);  
            
            /* Add the current point to the buffer or the profile */
            if (standard)

            /* Update variables */
            hiphase = lophase + deltaphase;
            lophase = hiphase - (int) hiphase;
            phase = phasenext;
        }

        

        // for (ii = onbin; ii <= offbin; ii++) 
        {
            int BlkPerRow = (nsub*(offbin-onbin+1)-1+512)/512;
            // data_bk = data[ii];
            // if (standard)
            //     add_to_prof<<<BlkPerRow,512>>>(prof, buffer, numprof, lophase_gpu, deltaphase_gpu, data, phaseadded, worklen);
            // else
                add_to_prof_sample<<<BlkPerRow,512>>>(prof, buffer, numprof, lophase_gpu, deltaphase_gpu, data, onbin, offbin, nsub, worklen);

            /* Use clever single pass mean and variance calculation */
            // stats->numdata += 1.0;
            // dev = data_bk - stats->data_avg;
            // stats->data_avg += dev / stats->numdata;
            // stats->data_var += dev * (data_bk - stats->data_avg);
        }

    } while (offbin < numdata - 1 && offbin != 0);

    /* Update and correct the statistics */
    // stats->prof_avg = 0.0;
    // for (ii = 0; ii < numprof; ii++)
    //     stats->prof_avg += prof[ii];
    // stats->prof_avg /= numprof;

    // /* Compute the Chi-Squared probability that there is a signal */
    // /* See Leahy et al., ApJ, Vol 266, pp. 160-170, 1983 March 1. */
    // stats->redchi = 0.0;
    // for (ii = 0; ii < numprof; ii++) {
    //     dtmp = prof[ii] - stats->prof_avg;
    //     stats->redchi += dtmp * dtmp;
    // }
    // stats->data_var /= (stats->numdata - 1.0);
    // stats->prof_var = stats->data_var * stats->numdata * profbinwidth;
    // stats->redchi /= (stats->prof_var * (numprof - 1));
    // phasenext = (phasenext < 0.0) ?
    //     1.0 + phasenext - (int) phasenext : phasenext - (int) phasenext;

    return (phasenext);
}

void Get_dmdelays_subband_int_gpu(double *search_dms_gpu, int *dmdelays, int numdmtrials, 
 int search_nsub,  double search_fold_p1, int search_proflen, double search_lofreq, 
 int search_numchan, double search_chan_wid, double search_avgvoverc)
{
    int BlkPerRow = (numdmtrials*search_nsub-1+256)/256;
    Get_dmdelays_subband_int_gpu_Do<<<BlkPerRow, 256>>>(search_dms_gpu, dmdelays, numdmtrials, 
  search_nsub,  search_fold_p1, search_proflen, search_lofreq, 
 search_numchan, search_chan_wid, search_avgvoverc);
}

__global__ void Get_dmdelays_subband_int_gpu_Do(double *search_dms_gpu, int *dmdelays, int numdmtrials, 
 int search_nsub,  double search_fold_p1, int search_proflen, double search_lofreq, 
 int search_numchan, double search_chan_wid, double search_avgvoverc)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=(numdmtrials*search_nsub)) return;

    int this_dm_id = MYgtid/search_nsub;
    int this_nsub = MYgtid - this_dm_id*search_nsub;

    double dm = search_dms_gpu[this_dm_id];

    double hif, dopplerhif, hifdelay, rdphase;

    rdphase = search_fold_p1 * search_proflen;
    hif = search_lofreq + (search_numchan - 1.0) * search_chan_wid;

    dopplerhif = hif * (1 + search_avgvoverc);
    if(dopplerhif == 0.0)
        hifdelay = 0.0;
    else 
        hifdelay = dm / (0.000241 * dopplerhif * dopplerhif);

        int chan_per_subband;
    double subbandwidth, losub_hifreq;

    chan_per_subband = search_numchan / search_nsub;
    subbandwidth = search_chan_wid * chan_per_subband;
    losub_hifreq = search_lofreq + subbandwidth - search_chan_wid;

    double freq;
    double delays;

    freq = (losub_hifreq + this_nsub * subbandwidth) * (1 + search_avgvoverc);
    if(dopplerhif == 0.0)
        delays = 0.0;
    else 
        delays = dm / (0.000241 * freq * freq);

    
    dmdelays[MYgtid] = NEAREST_INT((delays - hifdelay) * rdphase) % search_proflen;
}


void combine_subbands_1_gpu(double *inprofs,
                      int numparts, int numsubbands, int proflen, int numdmtrials,
                      int *delays, double *outprofs)
{
    int BlkPerRow = (numparts*proflen*numdmtrials-1+512)/512;
    combine_subbands_1_gpu_Do<<<BlkPerRow, 512>>>(inprofs,
                      numparts, numsubbands, proflen, numdmtrials,
                      delays, outprofs);
}


__global__ void combine_subbands_1_gpu_Do(double *inprofs,
                      int numparts, int numsubbands, int proflen, int numdmtrials,
                      int *delays, double *outprofs)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=(numparts*proflen*numdmtrials)) return;

    /* Combine the profiles */

    int jj, dm_id, part_id;

    dm_id = MYgtid / (numparts*proflen);  // dm_id
    part_id = (MYgtid - dm_id*numparts*proflen) / proflen;

    double tmp = 0.0;
    int delays_bk;
    int xx_id;
    for (jj = 0; jj < numsubbands; jj++) 
    {
        delays_bk = delays[jj + dm_id*numsubbands];
        xx_id = (MYgtid + delays_bk) % proflen;
        tmp += inprofs[part_id*numsubbands*proflen + jj*proflen + xx_id];
    }

    outprofs[MYgtid] = tmp;
}

void get_delays_gpu(double *delays_gpu, const int numpdds, const int numpds, const int  numps, const int npart, const int numtrials, const int proflen, double *fdotdots, double *fdots, const int good_ipd, const int good_ip, const int searchpddP, const int search_pstep, const long reads_per_part, const double proftime)
{
    int BlkPerRow = (numpdds*numpds*numps*npart-1+512)/512;
    get_delays_gpu_Do<<<BlkPerRow, 512>>>(delays_gpu, numpdds, numpds, numps, npart, numtrials, proflen, fdotdots, fdots, good_ipd, good_ip, searchpddP, search_pstep, reads_per_part, proftime);
}


__global__ void get_delays_gpu_Do(double *delays_gpu, const int numpdds, const int numpds, const int  numps, const int npart, const int numtrials, const int proflen, double *fdotdots, double *fdots, const int good_ipd, const int good_ip, const int searchpddP, const int search_pstep, const long reads_per_part, const double proftime)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=(numpdds*numpds*numps*npart)) return;

    int ii = MYgtid % npart;

    int ipdd  = MYgtid / (numpds*numps*npart);
    int ipd, ip;
    if(numpds == 1)
        ipd = good_ipd;
    else
        ipd = (MYgtid - ipdd*(numpds*numps*npart)) / (numps*npart);
    
    if(numps == 1)
        ip = good_ip;
    else
        ip = (MYgtid - ipdd*(numpds*numps*npart) - ipd* (numps*npart)) / npart;
    
    double fdotdots_ipdd = fdotdots[ipdd];
    double fdots_ipd = fdots[ipd];
    double parttimes_ii = ii * reads_per_part * proftime;
    double pdd_delays_ii = 0.0;
    double pd_delays_ii;

    double retval;
    if(searchpddP)
    {
        retval = fdotdots_ipdd * parttimes_ii * parttimes_ii * parttimes_ii /6.0;
        if(retval == -0)
            pdd_delays_ii = 0.0;
        else 
            pdd_delays_ii = retval * proflen;
    }
    
    retval = fdots_ipd * parttimes_ii * parttimes_ii /2.0;
    if(retval == -0)
        pd_delays_ii = pdd_delays_ii;
    else
        pd_delays_ii = pdd_delays_ii + retval* proflen;
    
    int totpdelay = search_pstep * (ip - (numtrials - 1) / 2);

    delays_gpu[MYgtid] = pd_delays_ii + (double)(ii*totpdelay) / npart;
}


//******************************************* normal loop  ***********************************************/

void combine_profs_1_gpu(double *profs, double *delays, int numprofs, int proflen, int numpdd, int numpd, int nump, double outstats_prof_avg, double outstats_prof_var, double *currentstats_redchi)
{
    int BlkPerRow = (nump*numpd*numpdd-1+512)/512;
    // combine_profs_1_gpu_Do<<<BlkPerRow, 512>>>(profs, delays, numprofs, proflen, numpdd, numtrials, outstats_prof_avg, outstats_prof_var, currentstats_redchi);
    // combine_profs_1_gpu_Do<<<BlkPerRow, 512, numprofs * sizeof(double)>>>(profs, delays, numprofs, proflen, numpdd, numpd, nump, outstats_prof_avg, outstats_prof_var, currentstats_redchi);
    combine_profs_1_gpu_Do<<<BlkPerRow, 512>>>(profs, delays, numprofs, proflen, numpdd, numpd, nump, outstats_prof_avg, outstats_prof_var, currentstats_redchi);
}

__global__ void combine_profs_1_gpu_Do(double *profs, double *delays, int numprofs, int proflen, int numpdd, int numpd, int nump, double outstats_prof_avg, double outstats_prof_var, double *currentstats_redchi)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=(nump*numpd*numpdd)) return;

    int ipdd = MYgtid/(nump*numpd);
    int ipd  = (MYgtid - ipdd*nump*numpd)/nump;
    int ip   = MYgtid - ipdd*nump*numpd - ipd*nump;

    int ii, jj, offset;
    
    /* Convert all the delays to positive offsets from   */
    /* the phase=0 profile bin, in units of profile bins */
    /* Note:  The negative sign refers to the fact that  */
    /*        we want positiev numbers to represent      */
    /*        shifts _to_ the right not _from_ the right */

    int offset_delay = ipdd*nump*numpd*numprofs + ipd*nump*numprofs + ip*numprofs;

    double outprof_bk, chixmeas = 0.0;
    for(jj=0; jj<proflen; jj++)
    {
        outprof_bk = 0.0;
        for (ii = 0; ii < numprofs; ii++)
        {
            // double local_delays_bk = fmod(-delays[ii+offset_delay], proflen*1.0);
            // if (local_delays_bk < 0.0)
            //     local_delays_bk += proflen;
            // /* Calculate the appropriate offset into the profile array */
            // offset = (int) (local_delays_bk + 0.5);
            // offset = (offset+jj)%proflen;

            // 优化后的 local_delays_bk 计算
            int int_delays = (int)(delays[ii+offset_delay] + 0.5);
            int local_delays_bk = (-int_delays) % proflen;
            if (local_delays_bk < 0) {
                local_delays_bk += proflen;
            }
            offset = (local_delays_bk+jj)%proflen;

            outprof_bk += profs[offset + ii*proflen];
        }
        outprof_bk = outprof_bk - outstats_prof_avg;
        chixmeas += outprof_bk*outprof_bk;
    }   
    currentstats_redchi[MYgtid] = chixmeas/outstats_prof_var/(proflen - 1.0);
}

/*-----------------------------------------------------------------------*
 * Fused DM-trial version of combine_profs_1_gpu: one launch covers all  *
 * DM trials.  Per (dm,trial) the arithmetic is identical (same double   *
 * operations in the same add order) to calling combine_profs_1_gpu once *
 * per DM, so the produced redchi values are bitwise identical.          *
 *-----------------------------------------------------------------------*/

__global__ void combine_profs_all_dms_gpu_Do(double *profs, double *delays, long long numdms, int numprofs, int proflen, int numpdd, int numpd, int nump, double outstats_prof_avg, double outstats_prof_var, double *currentstats_redchi, int redchi_stride, int ntrial, long long total)
{
    /* Grid-stride loop over TRIALS only.  Each thread precomputes the
     * integer per-bin shifts of its trial once (64 values) and then walks
     * every DM of the slab, so the (2 GB) delays array is read once per
     * trial instead of once per (dm, trial) cell.  The per-(dm,trial)
     * arithmetic is unchanged: bins accumulate in ii-ascending order and
     * the squared bins are summed in ascending bin order, bitwise equal to
     * combine_profs_1_gpu_Do. */
    const long long stride = (long long)gridDim.x * blockDim.x;
    const long long start = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    int shifts[128];

    for (long long trial = start; trial < ntrial; trial += stride) {
        const int ipdd = (int)(trial / (nump * numpd));
        const int ipd  = (int)((trial - ipdd * nump * numpd) / nump);
        const int ip   = (int)(trial - ipdd * nump * numpd - ipd * nump);
        const long long offset_delay = (long long)(ipdd * nump * numpd + ipd * nump + ip) * numprofs;

        if (proflen > 128) {
            /* heavy loop: cache shifts in local memory (proflen <= 1024 typical) */
            for (int ii = 0; ii < numprofs; ii++) {
                int ldel = (int)(delays[ii + offset_delay] + 0.5);
                ldel = (-ldel) % proflen;
                if (ldel < 0)
                    ldel += proflen;
                shifts[ii] = ldel;
            }
        }

        for (long long idm = 0; idm < numdms; idm++) {
            const double *profs_dm = profs + idm * numprofs * proflen;
            double chixmeas = 0.0;

            for (int b0 = 0; b0 < proflen; b0 += 16) {
                const int nb = (proflen - b0 < 16) ? (proflen - b0) : 16;
                double acc[16] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                for (int ii = 0; ii < numprofs; ii++) {
                    int ldel;
                    if (proflen > 128) {
                        ldel = shifts[ii];
                    } else {
                        ldel = (int)(delays[ii + offset_delay] + 0.5);
                        ldel = (-ldel) % proflen;
                        if (ldel < 0)
                            ldel += proflen;
                    }
                    const double *p = profs_dm + ii * proflen;
                    for (int j = 0; j < nb; j++) {
                        int k = ldel + b0 + j;
                        if (k >= proflen)
                            k -= proflen;
                        acc[j] += p[k];
                    }
                }
                for (int j = 0; j < nb; j++) {
                    double v = acc[j] - outstats_prof_avg;
                    chixmeas += v * v;
                }
            }
            currentstats_redchi[idm * redchi_stride + trial] = chixmeas / outstats_prof_var / (proflen - 1.0);
        }
    }
}

/* Per-DM argmax over the trial axis.  Ties keep the *first* (smallest)
 * index, matching thrust::max_element, which is what the previous
 * per-DM get_maxindex_double_gpu() calls used. */
void combine_profs_all_dms_gpu(double *profs, double *delays, int numdms, int numprofs, int proflen, int numpdd, int numpd, int nump, double outstats_prof_avg, double outstats_prof_var, double *temp_chix, double *currentstats_redchi, int redchi_stride)
{
    const int ntrial = nump * numpd * numpdd;
    const long long total = (long long)numdms * ntrial;
    /* cap the grid so huge trial sets never exceed launch limits */
    const int maxblocks = 4096;
    int blocks = (int)((total + 255) / 256);
    if (blocks > maxblocks)
        blocks = maxblocks;
    combine_profs_all_dms_gpu_Do<<<blocks, 256>>>(profs, delays, numdms, numprofs, proflen, numpdd, numpd, nump, outstats_prof_avg, outstats_prof_var, currentstats_redchi, redchi_stride, ntrial, total);
}

__global__ void get_maxindex_all_dms_gpu_Do(double *redchi, int numdms, int stride, double *maxV, int *maxindex)
{
    const int idm = blockIdx.x;
    if (idm >= numdms)
        return;
    const double *base = redchi + (long long)idm * stride;

    double best = base[0];
    int bestidx = 0;
    for (int i = 1 + threadIdx.x; i < stride; i += blockDim.x) {
        double v = base[i];
        if (v > best) {
            best = v;
            bestidx = i;
        }
    }

    __shared__ double sbest[256];
    __shared__ int sidx[256];
    sbest[threadIdx.x] = best;
    sidx[threadIdx.x] = bestidx;
    __syncthreads();
    for (int stride2 = blockDim.x >> 1; stride2 > 0; stride2 >>= 1) {
        if (threadIdx.x < stride2) {
            double ov = sbest[threadIdx.x + stride2];
            int oi = sidx[threadIdx.x + stride2];
            if (ov > sbest[threadIdx.x] || (ov == sbest[threadIdx.x] && oi < sidx[threadIdx.x])) {
                sbest[threadIdx.x] = ov;
                sidx[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        maxV[idm] = sbest[0];
        maxindex[idm] = sidx[0];
    }
}

void get_maxindex_all_dms_gpu(double *redchi, int numdms, int stride, double *maxV, int *maxindex)
{
    get_maxindex_all_dms_gpu_Do<<<numdms, 256>>>(redchi, numdms, stride, maxV, maxindex);
}






// *****************  good ********************** //
// void combine_profs_2_gpu(double *ddprofs_gpu, double *search_pdots_gpu, double *search_periods_gpu, float *parttimes_gpu, int numpdots, int numperiods, int npart, int search_proflen, double outstats_prof_avg, double outstats_prof_var, float *currentstats_redchi, double pfold, double search_fold_p2, double search_fold_p1, double *pdd_delays_gpu, float chifact)
// {
//     int BlkPerRow = (numpdots*numperiods-1+512)/512;
//     combine_profs_2_gpu_Do<<<BlkPerRow,512>>>(ddprofs_gpu, search_pdots_gpu, search_periods_gpu, parttimes_gpu, numpdots, numperiods, npart, search_proflen, outstats_prof_avg, outstats_prof_var, currentstats_redchi, pfold, search_fold_p2, search_fold_p1, pdd_delays_gpu, chifact);
// }

// __global__ void combine_profs_2_gpu_Do(double *ddprofs_gpu, double *search_pdots_gpu, double *search_periods_gpu, float *parttimes_gpu, int numpdots, int numperiods, int npart, int search_proflen, double outstats_prof_avg, double outstats_prof_var, float *currentstats_redchi, double pfold, double search_fold_p2, double search_fold_p1, double *pdd_delays_gpu, float chifact)
// {
//     const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
//     if(MYgtid>=(numpdots*numperiods)) return;

//     int ipd  = MYgtid/numperiods;
//     int ip   = MYgtid - ipd*numperiods;

//     double dfd = switch_pfdot_device(pfold, search_pdots_gpu[ipd]) - search_fold_p2;
//     double df = 1.0 / search_periods_gpu[ip] - search_fold_p1;


//     int ii, jj, offset;
//     double outprof_bk, chixmeas = 0.0;

//     for(jj=0; jj<search_proflen; jj++)
//     {
//         outprof_bk = 0.0;
//         for (ii = 0; ii < npart; ii++)
//         {
//             float parttimes_gpu_ii = parttimes_gpu[ii];
//             double pd_delays = pdd_delays_gpu[ii] + fdot2phasedelay_gpu(dfd, parttimes_gpu_ii);
//             double delays = (pd_delays + df * parttimes_gpu_ii) * search_proflen;

//             // double local_delays_bk = fmod(-delays, search_proflen*1.0);
//             // if (local_delays_bk < 0.0)
//             //         local_delays_bk += search_proflen;
//             // offset = (int) (local_delays_bk + 0.5);
//             // offset = (offset+jj)%search_proflen;

//             int int_delays = (int)(delays + 0.5);
//             int local_delays_bk = (-int_delays) % search_proflen;
//             if (local_delays_bk < 0) {
//                 local_delays_bk += search_proflen;
//             }
//             offset = (local_delays_bk+jj)%search_proflen;

//             outprof_bk += ddprofs_gpu[offset + ii*search_proflen];
//         }
//         outprof_bk = outprof_bk - outstats_prof_avg;
//         chixmeas += outprof_bk*outprof_bk;
//     }   
//     currentstats_redchi[MYgtid] = chixmeas/outstats_prof_var/(search_proflen - 1.0)*chifact;
// }


void combine_profs_2_gpu(double *ddprofs_gpu, double *search_pdots_gpu, double *search_periods_gpu, float *parttimes_gpu, int numpdots, int numperiods, int npart, int search_proflen, double outstats_prof_avg, double outstats_prof_var, float *currentstats_redchi, double pfold, double search_fold_p2, double search_fold_p1, double *pdd_delays_gpu, float chifact)
{
    int BlkPerRow = (numpdots*numperiods-1+512)/512;
    combine_profs_2_gpu_Do<<<BlkPerRow, 512, npart*(sizeof(float) + sizeof(double))>>>(ddprofs_gpu, search_pdots_gpu, search_periods_gpu, parttimes_gpu, numpdots, numperiods, npart, search_proflen, outstats_prof_avg, outstats_prof_var, currentstats_redchi, pfold, search_fold_p2, search_fold_p1, pdd_delays_gpu, chifact);
}

__global__ void combine_profs_2_gpu_Do(double *ddprofs_gpu, double *search_pdots_gpu, double *search_periods_gpu, float *parttimes_gpu, int numpdots, int numperiods, int npart, int search_proflen, double outstats_prof_avg, double outstats_prof_var, float *currentstats_redchi, double pfold, double search_fold_p2, double search_fold_p1, double *pdd_delays_gpu, float chifact)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ char shared_mem[];
    float *shared_parttimes = (float*)shared_mem;
    double *shared_pdd_delays = (double*)&shared_parttimes[npart];

    if(MYgtid>=(numpdots*numperiods)) return;

    int ipd  = MYgtid/numperiods;
    int ip   = MYgtid - ipd*numperiods;

    double dfd; // = switch_pfdot_device(pfold, search_pdots_gpu[ipd]) - search_fold_p2;
    double df = 1.0 / search_periods_gpu[ip] - search_fold_p1;
    
    double retval, pd_delays;
    float parttimes_gpu_ii;

    if (pfold == 0.0)
        dfd = -search_fold_p2;
    else
    {
        retval = -search_pdots_gpu[ipd] / (pfold * pfold);
        if (retval == -0)
            dfd = -search_fold_p2;
        else
            dfd = retval -search_fold_p2;
    }



    int ii, jj, offset;


    for(ii = threadIdx.x; ii < npart; ii += blockDim.x)
    {   
        shared_parttimes[ii] = parttimes_gpu[ii];
        shared_pdd_delays[ii] = pdd_delays_gpu[ii];
    }
    __syncthreads();

    double outprof_bk, chixmeas = 0.0;

    for(jj=0; jj<search_proflen; jj++)
    {
        outprof_bk = 0.0;
        for (ii = 0; ii < npart; ii++)
        {
            parttimes_gpu_ii = shared_parttimes[ii];
            retval = dfd * parttimes_gpu_ii * parttimes_gpu_ii / 2.0;
            if (retval == -0)
                pd_delays =  shared_pdd_delays[ii];
            else
                pd_delays =  retval + shared_pdd_delays[ii];

            double delays = (pd_delays + df * parttimes_gpu_ii) * search_proflen;

            // 优化后的 local_delays_bk 计算
            int int_delays = (int)(delays + 0.5);
            int local_delays_bk = (-int_delays) % search_proflen;
            if (local_delays_bk < 0) {
                local_delays_bk += search_proflen;
            }
            offset = (local_delays_bk+jj)%search_proflen;

            outprof_bk += ddprofs_gpu[offset + ii*search_proflen];
        }
        outprof_bk -= outstats_prof_avg;
        chixmeas += outprof_bk*outprof_bk;
    }   

    currentstats_redchi[MYgtid] = chixmeas/outstats_prof_var/(search_proflen - 1.0)*chifact;
}

// __global__ void combine_profs_3_gpu_Do(double *ddprofs_gpu, int nsub, int npart, int nproflen, double *delays_gpu, double *dmdelays_gpu, float *currentstats_redchi, double outstats_prof_avg, double outstats_prof_var)
// {
//     const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
//     if(MYgtid>=(nsub*nproflen)) return;

//     int id_sub  = MYgtid/nproflen;
//     int id_prof   = MYgtid - id_sub*nproflen;

//     int jj;
//     double dmdelays_gpu_bk = dmdelays_gpu[id_sub];
//     for(jj=0; jj<npart; jj++)
//     {
//         double totdelays = delays_gpu[jj] + dmdelays_gpu_bk;

//     }
// }

__device__ double switch_pfdot_device(double pf, double pfdot)
{
    double retval;

    if (pf == 0.0)
        return 0.0;
    else {
        retval = -pfdot / (pf * pf);
        if (retval == -0)
            return 0.0;
        else
            return retval;
    }
}


__device__ double fdot2phasedelay_gpu(double fdot, double time)
{
    double retval;

    retval = fdot * time * time / 2.0;
    if (retval == -0)
        return 0.0;
    else
        return retval;
}

__device__ float fdot2phasedelay_gpu_float(float fdot, float time)
{
    float retval;

    retval = fdot * time * time / 2.0;
    if (retval == -0)
        return 0.0;
    else
        return retval;
}

__device__ double calculate_delays(double pf, double pfdot, double search_fold_p2, double df, double time)
{
    double dfd = pf == 0.0 ? 0.0 : -pfdot / (pf * pf) - search_fold_p2;
    double pd_delays = dfd * time * time / 2.0;
    return (pd_delays + df * time);
}

//******************************************************************************************/

void get_redchi_gpu(double *currentstats_redchi, double *outprof,  double outstats_prof_avg, double outstats_prof_var, int proflen, int parts)
{
    int BlkPerRow = (parts -1+512)/512;
    get_redchi_gpu_Do<<<BlkPerRow,512>>>(currentstats_redchi, outprof,  outstats_prof_avg, outstats_prof_var, proflen, parts);
}

__global__ void get_redchi_gpu_Do(double *currentstats_redchi, double *outprof,  double outstats_prof_avg, double outstats_prof_var, int proflen, int parts)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=parts) return;


    int ii, jj;
    jj = MYgtid/proflen;

    double dtmp, chitmp, chixmeas = 0.0;
    for (ii = 0; ii < proflen; ii++) {
        dtmp = outprof[ii+ jj+proflen];
        chitmp = dtmp - outstats_prof_avg;
        chixmeas += (chitmp * chitmp);
    }
    currentstats_redchi[MYgtid] = chixmeas / outstats_prof_var / (proflen - 1.0);
}




void Set_cufftComplex_date_as_zero_gpu(fcomplex *data, long long num)
{

    Set_cufftComplex_date_as_zero_gpu_Do<<<(num+255)/256, 256>>>(data, num);
}

__global__ void Set_cufftComplex_date_as_zero_gpu_Do(fcomplex *data, long long num)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=num) return;

    data[idx].r = 0.0f;
    data[idx].i = 0.0f;
}

void compute_power_gpu(fcomplex *data, float *powers, int numdata) 
{
    compute_power_kernel<<<(numdata+255)/256, 256>>>(data, powers, numdata);
}


__global__ void compute_power_kernel(fcomplex *data, float *powers, int numdata) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=numdata) return;

    float  real = data[idx].r;
    float  imag = data[idx].i;
    powers[idx] = real * real + imag * imag;
}


void sort_and_get_median_gpu(float *data, int numdata, float *median) 
{
    thrust::device_ptr<float> dev_ptr(data);
    thrust::sort(dev_ptr, dev_ptr + numdata);


    int middle = (numdata-1) / 2;
    cudaMemcpy(median, data + middle, sizeof(float), cudaMemcpyDeviceToHost);
}



/*----------------------------------------------------------------------*
 * Chunk-batched kernels (process K consecutive r-chunks in one launch).*
 * Used by accelsearch and accelsearchlist1 to reduce the number of     *
 * tiny kernel launches and to feed cuFFT with larger batches.          *
 *----------------------------------------------------------------------*/

void spread_no_pad_gpu_dat_chunks(cufftComplex * data, int numdata,
                                  cufftComplex * result, int numresult,
                                  int numbetween, double fullrlo, double rstep,
                                  double harm_fract, int binoffset,
                                  long long obs_lobin, int readdatanum)
{
    int BlkPerRow = (numdata * readdatanum - 1 + 256) / 256;
    Do_spread_with_pad_GPU_dat_chunks<<<BlkPerRow, 256>>>(data, numdata,
        result, numresult, numbetween, fullrlo, rstep, harm_fract,
        binoffset, obs_lobin, readdatanum);
}

__global__ void Do_spread_with_pad_GPU_dat_chunks(const cufftComplex *data,
    int numdata, cufftComplex *result, int numresult, int numbetween,
    double fullrlo, double rstep, double harm_fract, int binoffset,
    long long obs_lobin, int readdatanum)
{
    const int MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if (MYgtid >= numdata * readdatanum)
        return;
    const int c = MYgtid / numdata;
    const int j = MYgtid - c * numdata;

    /* Replicate the host-side per-chunk index math (identical double ops) */
    const double fullrlo_c = fullrlo + (double)c * rstep;
    const double drlo = rint(ACCEL_RDR * fullrlo_c * harm_fract) * ACCEL_DR;
    const long long rlo_c = (long long) floor(drlo);
    const long long lobin = rlo_c - binoffset;
    long long offset = 0;
    if (lobin - obs_lobin < 0)
        offset = llabs(lobin - obs_lobin);
    const long long firstbin = (lobin - obs_lobin) + offset;

    const float norm = c_norm_data[c];
    cufftComplex *res = result + (long long)c * numresult;

    if (j < offset) {
        res[j].x = 0.0f;
        res[j].y = 0.0f;
    } else {
        const float2 data_bk = __ldg(&data[firstbin + j]);
        res[j * numbetween].x = data_bk.x * norm;
        res[j * numbetween].y = data_bk.y * norm;
        res[j * numbetween + 1].x = 0.0f;
        res[j * numbetween + 1].y = 0.0f;
    }
}


void add_subharm_gpu_chunkl_allharm(float *powers_out, cufftComplex *fdata,
    unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0,
    int K, int outpows_gpu_xlen, int output_x_max, float powcut, int numharm,
    int *cand_num_gpu, accel_cand_gpu * cand_array_search_gpu, int fdata_len,
    int obs_corr_uselen, int obs_numz, int stage_base, int nsub)
{
    int BlkPerRow = (numrs_0 * numzs_0 * K - 1 + 256) / 256;
    Do_add_subharm_gpu_chunkl_allharm<<<BlkPerRow, 256>>>(powers_out, fdata,
        rinds, zinds, numrs_0, numzs_0, K, outpows_gpu_xlen, output_x_max,
        powcut, numharm, cand_num_gpu, cand_array_search_gpu, fdata_len,
        obs_corr_uselen, obs_numz, stage_base, nsub);
}

__global__ void Do_add_subharm_gpu_chunkl_allharm(float *powers_out,
    cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds,
    int numrs_0, int numzs_0, int K, int outpows_gpu_xlen, int output_x_max,
    float powcut, int numharm, int *d_addr,
    accel_cand_gpu * cand_array_search_gpu, int fdata_len,
    int obs_corr_uselen, int obs_numz, int stage, int nsub)
{
    const long long MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if (MYgtid >= (long long)numrs_0 * numzs_0 * K)
        return;

    const long long dataid = MYgtid / (numrs_0 * numzs_0);
    const long long yy = (MYgtid - dataid * numrs_0 * numzs_0) / numrs_0;
    const int xx = (int)(MYgtid % numrs_0);

    const long long rinds_base = dataid * (long long)nsub * obs_corr_uselen;
    const long long zinds_base = dataid * (long long)nsub * obs_numz;

    int ii;
    float total_power = 0.0f;

    #pragma unroll
    for (ii = 0; ii < numharm / 2; ii++) {
        const gpu_parameter_add gpu_parameter_add_tmp =
            c_gpu_parameter_add[ii + numharm / 2];
        const unsigned short addr_r =
            __ldg(&rinds[rinds_base + (stage + ii) * obs_corr_uselen + xx]);
        const unsigned short addr_z =
            __ldg(&zinds[zinds_base + (stage + ii) * obs_numz + yy]);
        const long long addr_result = dataid * gpu_parameter_add_tmp.fftlen *
            gpu_parameter_add_tmp.numkern + addr_z * gpu_parameter_add_tmp.fftlen
            + addr_r + gpu_parameter_add_tmp.offset;
        const float2 fdata_b = __ldg(&fdata[addr_result + ii * fdata_len]);
        total_power += (fdata_b.x * fdata_b.x + fdata_b.y * fdata_b.y) *
            gpu_parameter_add_tmp.norm_data;
    }

    const long long outpows_addr = dataid * outpows_gpu_xlen + yy * numrs_0 + xx;
    total_power += powers_out[outpows_addr];
    powers_out[outpows_addr] = total_power;

    {
        bool is_cand = (total_power > powcut);
        accel_cand_gpu cand_tmp;
        if (is_cand) {
            cand_tmp.pow = total_power;
            cand_tmp.numharm = numharm;
            cand_tmp.z_ind = (int)yy;
            cand_tmp.r_ind = xx;
        }

        /* Count/stash candidates per-chunk with a plain per-thread atomic.
         * Do NOT aggregate over a warp here: when a warp spans two chunks
         * (different 'dataid' values within one warp), the old ballot path
         * bumped only lane 0's d_addr[dataid] but stored candidates using
         * every lane's own dataid.  That left d_addr[] counts inconsistent
         * with what was actually written, so the host copied uninitialized
         * candidate slots (numharm = 0, pow = 0) and later died in
         * calc_props() with "data.pow = 0.000000 (out-of-bounds)". */
        if (is_cand) {
            const int addr_search = atomicAdd(&d_addr[dataid], 1);
            cand_array_search_gpu[addr_search + dataid * output_x_max] = cand_tmp;
        }
    }
}

/*----------------------------------------------------------------------*
 * 2026-batch optimization: GPU-side index generation, median/normal    *
 * factor generation, and global-norm spread variants.                  *
 *                                                                      *
 * Every kernel below replicates, operation for operation (same IEEE    *
 * double ops in the same order, same round-to-nearest casts), the      *
 * host code it replaces, so results are bit-identical to the CPU       *
 * computed tables/norms.  The power-per-bin value is computed as       *
 * fmaf(r, r, i*i) which matches the GCC -Ofast FMA contraction.        *
 *----------------------------------------------------------------------*/

/* Per-harmonic-unit descriptor: one entry per subharm_fderivs call
 * (the fundamental plus every (stage, harm) pair, in call order).     */
static mednorm_desc *g_mednorm_descs = NULL;
static int           g_mednorm_descs_n = 0;
static double       *g_rind_fracts = NULL;
static int           g_rind_fracts_n = 0;

void mednorm_set_descs(const mednorm_desc *h, int n)
{
    if (!g_mednorm_descs || g_mednorm_descs_n < n) {
        if (g_mednorm_descs) cudaFree(g_mednorm_descs);
        cudaMalloc((void **)&g_mednorm_descs, sizeof(mednorm_desc) * n);
    }
    cudaMemcpy(g_mednorm_descs, h, sizeof(mednorm_desc) * n,
               cudaMemcpyHostToDevice);
    g_mednorm_descs_n = n;
}

void rind_set_fracts(const double *h, int n)
{
    if (!g_rind_fracts || g_rind_fracts_n < n) {
        if (g_rind_fracts) cudaFree(g_rind_fracts);
        cudaMalloc((void **)&g_rind_fracts, sizeof(double) * n);
    }
    cudaMemcpy(g_rind_fracts, h, sizeof(double) * n, cudaMemcpyHostToDevice);
    g_rind_fracts_n = n;
}

/* rinds for K consecutive chunks.  out[((c * nsub) + kk) * corr_uselen
 * + ii], matching the host loop order (kk over the (stage, harm) pairs
 * of stages 1..numharmstages-1). */
__global__ void rind_gen_kernel(const double * __restrict__ fracts,
                                unsigned short * __restrict__ rinds,
                                double fullrlo, double rstep,
                                int corr_uselen, int nsub, int K, int nfract)
{
    const long long g = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (g >= (long long)K * nsub * corr_uselen) return;
    const int ii = (int)(g % corr_uselen);
    const long long t = g / corr_uselen;
    const int kk = (int)(t % nsub);
    if (kk >= nfract) return;
    const int c = (int)(t / nsub);

    const double harm_fract = fracts[kk];
    const double fullrlo_c = fullrlo + (double)c * rstep;
    const double drlo = rint(ACCEL_RDR * fullrlo_c * harm_fract) * ACCEL_DR;
    const long long rlo = (long long) floor(drlo);
    const double rr = fullrlo_c + ii * ACCEL_DR;
    const double subr = rint(ACCEL_RDR * rr * harm_fract) * ACCEL_DR;
    rinds[g] = (unsigned short)((int)((subr - (double)rlo) * ACCEL_RDR + DBLCORRECT));
}

void rind_gen_chunks_gpu(unsigned short *rinds, double fullrlo,
                         int corr_uselen, int nsub, int K)
{
    const double rstep = (double)corr_uselen * ACCEL_DR;
    const long long total = (long long)K * nsub * corr_uselen;
    int BlkPerRow = (int)((total - 1 + 256) / 256);
    rind_gen_kernel<<<BlkPerRow, 256>>>(g_rind_fracts, rinds, fullrlo,
                                        rstep, corr_uselen, nsub, K,
                                        g_rind_fracts_n);
}

/* zinds, chunk-independent.  out[((c * nsub) + kk) * numz + ii]; the K
 * chunk slots hold identical copies, as the host path did. */
__global__ void zind_gen_kernel(const double * __restrict__ fracts,
                                unsigned short * __restrict__ zinds,
                                double obszlo, int numz, int nsub, int K,
                                int nfract)
{
    const long long g = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (g >= (long long)K * nsub * numz) return;
    const int ii = (int)(g % numz);
    const long long t = g / numz;
    const int kk = (int)(t % nsub);
    if (kk >= nfract) return;
    const int c = (int)(t / nsub);
    (void)c;

    const double harm_fract = fracts[kk];
    const double zlo_d = NEAREST_INT(ACCEL_RDZ * obszlo * harm_fract) * ACCEL_DZ;
    const long long zlo = (long long) floor(zlo_d);
    const double zz = obszlo + ii * ACCEL_DZ;
    const double subz = NEAREST_INT(ACCEL_RDZ * zz * harm_fract) * ACCEL_DZ;
    zinds[g] = (unsigned short)((int)((subz - (double)zlo) * ACCEL_RDZ + DBLCORRECT));
}

void zind_gen_chunks_gpu(unsigned short *zinds, double obszlo, int numz,
                         int nsub, int K)
{
    const long long total = (long long)K * nsub * numz;
    int BlkPerRow = (int)((total - 1 + 256) / 256);
    zind_gen_kernel<<<BlkPerRow, 256>>>(g_rind_fracts, zinds, obszlo,
                                        numz, nsub, K, g_rind_fracts_n);
}

/* Median power + normalization factor for every (harm unit, chunk)
 * segment of a batch.  One block per segment; bitonic sort on the
 * zero-padded power array, the (numdata-1)/2-th order statistic is
 * exactly the value the host quickselect returned, and the norm is
 * computed in the same double ops. */
#define MEDNORM_THREADS 512
#define MEDNORM_MAXN 4096

__global__ void mednorm_gen_kernel(const cufftComplex * __restrict__ fft,
                                   long long frame_stride,
                                   long long base_pre,
                                   const mednorm_desc * __restrict__ descs,
                                   double fullrlo, double rstep,
                                   int K, long long obs_lobin,
                                   long long obs_numbins, int ncalls,
                                   float * __restrict__ norm_out)
{
    __shared__ float sarr[MEDNORM_MAXN];
    __shared__ long long sh_firstbin, sh_offset, sh_numpad;

    const int seg = blockIdx.x;
    if (seg >= ncalls * K) return;
    const int s = seg / K;
    const int c = seg - s * K;
    const mednorm_desc d = descs[s];
    const int numdata = d.fftlen / ACCEL_NUMBETWEEN;

    if (threadIdx.x == 0) {
        const double fullrlo_c = fullrlo + (double)c * rstep;
        const double drlo = rint(ACCEL_RDR * fullrlo_c * d.harm_fract) * ACCEL_DR;
        const long long rlo = (long long) floor(drlo);
        const long long lobin = rlo - d.binoffset;
        long long offset = 0;
        if (lobin - obs_lobin < 0)
            offset = llabs(lobin - obs_lobin);
        const long long firstbin = (lobin - obs_lobin) + offset;
        long long newnumbins = numdata - offset;
        long long numpad = 0;
        if (firstbin + newnumbins > obs_numbins) {
            numpad = firstbin + newnumbins - obs_numbins;
            newnumbins -= numpad;
        }
        sh_firstbin = firstbin;
        sh_offset = offset;
        sh_numpad = numpad;
    }
    __syncthreads();
    const long long firstbin = sh_firstbin;
    const long long offset = sh_offset;
    const long long numpad = sh_numpad;
    const long long hi = numdata - numpad;   /* j >= hi -> 0 */
    /* powers[offset + jj] = |fft[firstbin + jj]|^2, jj in [0,newnumbins) */
    const cufftComplex *segp = fft + base_pre + (long long)c * frame_stride
                               + firstbin - offset;

    for (int j = threadIdx.x; j < numdata; j += MEDNORM_THREADS) {
        float p = 0.0f;
        if (j >= offset && j < hi) {
            const float2 v = __ldg(&segp[j]);
            p = fmaf(v.x, v.x, __fmul_rn(v.y, v.y));
        }
        sarr[j] = p;
    }
    /* next pow2 >= numdata with +INF sentinels */
    int P = 1;
    while (P < numdata) P <<= 1;
    for (int j = numdata + threadIdx.x; j < P; j += MEDNORM_THREADS)
        sarr[j] = CUDART_INF_F;
    __syncthreads();

    for (int k = 2; k <= P; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < P; i += MEDNORM_THREADS) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const float a = sarr[i];
                    const float b = sarr[ixj];
                    const bool up = ((i & k) == 0);
                    if ((a > b) == up) {
                        sarr[i] = b;
                        sarr[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        const float med = sarr[(numdata - 1) / 2];
        norm_out[seg] = (float)(1.0 / sqrt((double)med / 0.69314718055994530941723));
    }
}

void mednorm_gen_gpu(const cufftComplex *fft, long long frame_stride,
                     long long base_pre, double fullrlo, double rstep,
                     int K, long long obs_lobin, long long obs_numbins,
                     int ncalls, float *norm_out)
{
    mednorm_gen_kernel<<<ncalls * K, MEDNORM_THREADS>>>(fft, frame_stride,
        base_pre, g_mednorm_descs, fullrlo, rstep, K, obs_lobin,
        obs_numbins, ncalls, norm_out);
}

/* Spread variants that read the per-chunk normalization factors from a
 * device array instead of __constant__ memory.  Bit-identical output:
 * same expression (data * norm), same layout. */
void spread_no_pad_gpu_dat_chunks_gn(cufftComplex * data, int numdata,
                                     cufftComplex * result, int numresult,
                                     int numbetween, double fullrlo,
                                     double rstep, double harm_fract,
                                     int binoffset, long long obs_lobin,
                                     int readdatanum, const float *norms)
{
    int BlkPerRow = (numdata * readdatanum - 1 + 256) / 256;
    Do_spread_with_pad_GPU_dat_chunks_gn<<<BlkPerRow, 256>>>(data, numdata,
        result, numresult, numbetween, fullrlo, rstep, harm_fract,
        binoffset, obs_lobin, readdatanum, norms);
}

__global__ void Do_spread_with_pad_GPU_dat_chunks_gn(const cufftComplex *data,
    int numdata, cufftComplex *result, int numresult, int numbetween,
    double fullrlo, double rstep, double harm_fract, int binoffset,
    long long obs_lobin, int readdatanum, const float * __restrict__ norms)
{
    const int MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if (MYgtid >= numdata * readdatanum)
        return;
    const int c = MYgtid / numdata;
    const int j = MYgtid - c * numdata;

    const double fullrlo_c = fullrlo + (double)c * rstep;
    const double drlo = rint(ACCEL_RDR * fullrlo_c * harm_fract) * ACCEL_DR;
    const long long rlo_c = (long long) floor(drlo);
    const long long lobin = rlo_c - binoffset;
    long long offset = 0;
    if (lobin - obs_lobin < 0)
        offset = llabs(lobin - obs_lobin);
    const long long firstbin = (lobin - obs_lobin) + offset;

    const float norm = norms[c];
    cufftComplex *res = result + (long long)c * numresult;

    if (j < offset) {
        res[j].x = 0.0f;
        res[j].y = 0.0f;
    } else {
        const float2 data_bk = __ldg(&data[firstbin + j]);
        res[j * numbetween].x = data_bk.x * norm;
        res[j * numbetween].y = data_bk.y * norm;
        res[j * numbetween + 1].x = 0.0f;
        res[j * numbetween + 1].y = 0.0f;
    }
}

void spread_no_pad_gpu_dat_list_loiops_gn(cufftComplex * data, int numdata,
    cufftComplex *result, int numresult, int numbetween,
    long long offset_bk, long long numpad_bk, long long newnumbins_bk,
    long long addbins, int readdatanum, int data_len, const float *norms)
{
    int BlkPerRow = (numresult*readdatanum-1+256)/256;
    Do_spread_with_pad_GPU_dat_list_loiops_gn<<<BlkPerRow, 256>>>(data,
        numdata, result, numresult, numbetween, offset_bk, addbins,
        readdatanum, data_len, norms);
}

__global__ void Do_spread_with_pad_GPU_dat_list_loiops_gn(
    const cufftComplex *data, int numdata, cufftComplex *result,
    int numresult, int numbetween, long long offset_bk, long long addbins,
    int readdatanum, int data_len, const float * __restrict__ norms)
{
    const long long MYgtid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (MYgtid >= (long long)numresult * readdatanum) return;

    const long long k = MYgtid / numbetween;
    if (MYgtid % numbetween == 0 && k < (long long)numdata * readdatanum) {
        const int id_x = (int)(k % numdata);
        const int id_y = (int)(k / numdata);
        const float norm_bk = norms[id_y];
        const float2 data_bk = __ldg(&data[(long long)id_y * data_len + id_x + addbins]);
        result[MYgtid].x = data_bk.x * norm_bk;
        result[MYgtid].y = data_bk.y * norm_bk;
    } else {
        result[MYgtid].x = 0.0f;
        result[MYgtid].y = 0.0f;
    }
}


/* float4-widened variant of Do_loops_in_GPU_1_list: identical
 * per-element expressions (same results, wider loads/stores). */
__global__ void Do_loops_in_GPU_1_list_f4(const cufftComplex * __restrict__ fpdata,
    const cufftComplex * __restrict__ fkern, cufftComplex * __restrict__ outdata,
    int fftlen, int numzs, int readdatanum)
{
    const long long total = (long long)fftlen * numzs * readdatanum;
    const long long gid2 = 2LL * ((long long)blockDim.x * blockIdx.x + threadIdx.x);

    const long long zspan = (long long)fftlen * numzs;
    const long long g0 = gid2;
    if (g0 >= total)
        return;

    const int xnum0 = (int)(g0 % fftlen);
    if (xnum0 == fftlen - 1 || gid2 + 1 >= total) {
        /* boundary element: scalar path */
        const long long dataid = g0 / zspan;
        const long long zid = (g0 - dataid * zspan) / fftlen;
        const float dr = __ldg(&fpdata[xnum0 + dataid * fftlen].x);
        const float di = __ldg(&fpdata[xnum0 + dataid * fftlen].y);
        const float kr = __ldg(&fkern[xnum0 + zid * fftlen].x);
        const float ki = __ldg(&fkern[xnum0 + zid * fftlen].y);
        cufftComplex o;
        o.x = dr * kr + di * ki;
        o.y = di * kr - dr * ki;
        outdata[g0] = o;
        return;
    }

    const long long dataid = g0 / zspan;
    const long long zid = (g0 - dataid * zspan) / fftlen;

    const float4 d4 = __ldg(reinterpret_cast<const float4 *>(&fpdata[xnum0 + dataid * fftlen]));
    const float4 k4 = __ldg(reinterpret_cast<const float4 *>(&fkern[xnum0 + zid * fftlen]));

    cufftComplex o0, o1;
    o0.x = d4.x * k4.x + d4.y * k4.y;
    o0.y = d4.y * k4.x - d4.x * k4.y;
    o1.x = d4.z * k4.z + d4.w * k4.w;
    o1.y = d4.w * k4.z - d4.z * k4.w;

    float4 out4;
    out4.x = o0.x; out4.y = o0.y; out4.z = o1.x; out4.w = o1.y;
    outdata[g0]     = o0;
    outdata[g0 + 1] = o1;
    (void)out4;
}

void loops_in_GPU_1_list_f4(cufftComplex *fpdata, cufftComplex *fkern, cufftComplex *outdata, int fftlen, int numzs, int readdatanum)
{
    const long long total = (long long)fftlen * numzs * readdatanum;
    int BlkPerRow = (int)((total / 2 + total % 2 - 1 + 256) / 256);
    Do_loops_in_GPU_1_list_f4<<<BlkPerRow, 256>>>(fpdata, fkern, outdata, fftlen, numzs, readdatanum);
}
