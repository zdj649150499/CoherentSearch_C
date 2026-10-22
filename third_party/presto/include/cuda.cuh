#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <cuda.h>
#include <cufft.h>


#ifdef __cplusplus
extern "C" {
#endif


#define ACCEL_USELEN 7470 // This works up to zmax=300 to use 8K FFTs
// #define ACCEL_USELEN 7960 // This works up to zmax=100 to use 8K FFTs
// #define ACCEL_USELEN 3850 // This works up to zmax=100 to use 4K FFTs
// #define ACCEL_USELEN 1820 // This works up to zmax=100 to use 2K FFTs

/* Stepsize in Fourier Freq */
#define ACCEL_NUMBETWEEN 2
/* Stepsize in Fourier Freq */
#define ACCEL_DR  0.5
/* Reciprocal of ACCEL_DR */
#define ACCEL_RDR 2
/* Stepsize in Fourier F-dot */
#define ACCEL_DZ  2
/* Stepsize in Fourier F-dot-dot */
#define ACCEL_DW 20
/* Reciprocal of ACCEL_DW */
#define ACCEL_RDW 0.05
/* Closest candidates we will accept as independent */
#define ACCEL_CLOSEST_R 15.0
/* Padding for .dat file reading so that we don't SEGFAULT */
#define ACCEL_PADDING 2000



#define CUDA_CHECK(call)                            \
do                                                  \
{                                                   \
    const cudaError_t error_code = call;            \
    if (error_code != cudaSuccess)                  \
    {                                               \
        printf("CUDA Error:\n");                    \
        printf("    File:       %s\n", __FILE__);   \
        printf("    Line:       %d\n", __LINE__);   \
        printf("    Error code: %d\n", error_code); \
        printf("    Error text: %s\n",              \
                cudaGetErrorString(error_code));    \
        exit(1);                                    \
    }                                               \
} while (0)




#ifndef _FCOMPLEX_DECLARED_
typedef struct fcomplex {
    float r, i;
} fcomplex;
#define _FCOMPLEX_DECLARED_
#endif				/* _FCOMPLEX_DECLARED_ */

typedef struct accel_cand_gpu{
	float pow;					/*pow of selected candidate*/
    int   numharm;
	// int		nof_cand;			/*number of candidates in sub_array/plane */
	int		z_ind;				/*z_index of the selected candidate*/
	int		r_ind;				/*r_index of the selected candidate*/
    // int		w_ind;				/*w_index of the selected candidate*/
}	accel_cand_gpu;

typedef struct foldstats_gpu {
  double numdata;     /* Number of data bins folded         */
  double data_avg;    /* Average level of the data bins     */
  double data_var;    /* Variance of the data bins          */
  double numprof;     /* Number of bins in the profile      */
  double prof_avg;    /* Average level of the profile bins  */
  double prof_var;    /* Variance of the profile bins       */
  double redchi;      /* Reduced chi-squared of the profile */
} foldstats_gpu;

typedef struct POSITION_gpu {
    float pow;	    /* Power normalized with local power             */
    double p1;	    /* r if rzw search, r_startfft if bin search     */
    double p2;	    /* z if rzw search, r_freqmod if bin search      */
    double p3;	    /* w if rzw search, numfftbins if bin search     */
} position_gpu;

typedef struct orbitparams_gpu {
    double p;	    /* Orbital period (s)                            */
    double e;	    /* Orbital eccentricity                          */
    double x;	    /* Projected semi-major axis (lt-sec)            */
    double w;	    /* Longitude of periapsis (deg)                  */
    double t;	    /* Time since last periastron passage (s)        */
    double pd;	    /* Orbital period derivative (s/yr)              */
    double wd;	    /* Advance of longitude of periapsis (deg/yr)    */
} orbitparams_gpu;

typedef struct PREPFOLDINFO_GPU {
  double *rawfolds;   /* Raw folds (nsub * npart * proflen points) */
  double *dms;        /* DMs used in the trials */
  double *periods;    /* Periods used in the trials */
  double *pdots;      /* P-dots used in the trials */
  foldstats_gpu *stats;   /* Statistics for the raw folds */
  int numdms;         /* Number of 'dms' */
  int numperiods;     /* Number of 'periods' */
  int numpdots;       /* Number of 'pdots' */
  int nsub;           /* Number of frequency subbands folded */
  int npart;          /* Number of folds in time over integration */
  int proflen;        /* Number of bins per profile */
  int numchan;        /* Number of channels for radio data */
  int pstep;          /* Minimum period stepsize in profile phase bins */
  int pdstep;         /* Minimum p-dot stepsize in profile phase bins */
  int dmstep;         /* Minimum DM stepsize in profile phase bins */
  int ndmfact;        /* 2*ndmfact*proflen+1 DMs to search */
  int npfact;         /* 2*npfact*proflen+1 periods and p-dots to search */
  char *filenm;       /* Filename of the folded data */
  char *candnm;       /* String describing the candidate */
  char *telescope;    /* Telescope where observation took place */
  char *pgdev;        /* PGPLOT device to use */
  char rastr[16];     /* J2000 RA  string in format hh:mm:ss.ssss */
  char decstr[16];    /* J2000 DEC string in format dd:mm:ss.ssss */
  double dt;          /* Sampling interval of the data */
  double startT;      /* Fraction of observation file to start folding */
  double endT;        /* Fraction of observation file to stop folding */
  double tepoch;      /* Topocentric eopch of data in MJD */
  double bepoch;      /* Barycentric eopch of data in MJD */
  double avgvoverc;   /* Average topocentric velocity */
  double lofreq;      /* Center of low frequency radio channel */
  double chan_wid;    /* Width of each radio channel in MHz */
  double bestdm;      /* Best DM */
  position_gpu topo;      /* Best topocentric p, pd, and pdd */
  position_gpu bary;      /* Best barycentric p, pd, and pdd */
  position_gpu fold;      /* f, fd, and fdd used to fold the initial data */
  orbitparams_gpu orb;    /* Barycentric orbital parameters used in folds */
} prepfoldinfo_gpu;

typedef struct gpu_parameter_add { 
    int offset;
    int fftlen;
    float norm_data;
    int numkern;
} gpu_parameter_add;



void cp_gpu_parameter_add_list_to_const(gpu_parameter_add *gpu_parameter_add_list_cpu, int num);
void cp_norm_data_to_const(float *norm_data, int num);

void select_cuda_dev(int cuda_inds);
void Endup_GPU();

unsigned int calc_next_32_times(unsigned int x);
// int configure_reduce_sum(dim3 &grid_dim, dim3 &block_dim, unsigned int next_pow_32_times);


float get_mean_gpu(float *indata, int N);
double get_mean_gpu_d(double *indata, int N);
float get_variance_gpu(float *indata, float mean,int N);
void get_maxindex_double_gpu(double *indata, int N, double *maxV, int *maxindex);

void sum_value_gpu(float *indata, float value,int N);
void sum_value_gpu_stream(float *indata, double *mean,int N, int M, cudaStream_t stream_1, cudaStream_t stream_2);
__global__ void Do_sum_value_gpu(float *indata, float value,int N);

float gpu_sum_reduce_float(float *arr, int total_num);
__global__ void block_sum_reduce_float(float *g_idata, float *g_odata, unsigned int n, int blockSize);

float gpu_varience_reduce_float(float *arr, float mean, int total_num);
__global__ void block_varience_reduce_float(float *g_idata, double *g_odata, float mean, unsigned int n, int blockSize) ;

void mask_data_GPU(float *currentdata_gpu, int *maskchans_gpu, float *padvals_gpu, int spectra_per_subint, int num_channels, int nummasked);
__global__ void Do_mask_data_GPU(float *currentdata_gpu, int *maskchans_gpu, float *padvals_gpu, int spectra_per_subint, int num_channels, int nummasked);

void ignorechans_GPU(float *currentdata_gpu, int *ignorechans_gpu, int spectra_per_subint, int num_channels, int num_ignorechans);
__global__ void DO_ignorechans_GPU(float *currentdata_gpu, int *ignorechans_gpu, int spectra_per_subint, int num_channels, int num_ignorechans);

void dedisp_subbands_GPU(float *lastdata_gpu,float *lastdata,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose);
void dedisp_subbands_GPU_cache(unsigned char *data_gpu, float *data_gpu_scl, float *data_gpu_offs, unsigned char *lastdata_gpu, float *lastdata_gpu_scl, float *lastdata_gpu_offs,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose);
void dedisp_subbands_GPU_cache_stream(unsigned char *data_gpu, float *data_gpu_scl, float *data_gpu_offs, unsigned char *lastdata_gpu, float *lastdata_gpu_scl, float *lastdata_gpu_offs,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose, cudaStream_t stream);
__global__ void Do_dedisp_subbands_GPU(float *data, float *lastdata,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose);
__global__ void Do_dedisp_subbands_GPU_cache(unsigned char *data, float *data_scl, float *data_offs, unsigned char *lastdata, float *lastdata_scl, float *lastdata_offs,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose);
void downsamp_GPU(float *indata, float *outdata, int numchan, int numpts, int down, int transpose);
__global__ void Do_downsamp_GPU(float *indata, float *outdata, int numchan, int numpts, int down, int transpose);

void float_dedisp_GPU(float *data, float *lastdata,
                  int numpts, int numchan,
                  int *delays, float approx_mean, float *result, int numdms, int transpose);
__global__ void Do_float_dedisp_GPU(float *data, float *lastdata,
                  int numpts, int numchan,
                  int *delays, float approx_mean, float *result, int numdms, int transpose);

void Get_subsdata(float *indata, short *outdata, int nsub, int worklen);
__global__ void Do_Get_subsdata(float *indata, short *outdata, int nsub, int worklen);

void SetSubData4Mask_GPU(float *subbanddata, float *padvals, int *maskchans,int len, int Width);
__global__ void Do_SetSubData4Mask_GPU(float *subbanddata, float *padvals, int *maskchans, int len, int Width);

void ZeroDM_subchan_GPU(float *indata, int len, int Width);
__global__ void Add_data_GPU(float *indata, float value, int len);

int iDivUp(int a, int b);

void transpose_GPU(float *d_src, float *d_dest, int width, int height);
void transpose_GPU_short(short *d_src, float *d_dest, int width, int height);
__global__ void d_transpose(float *odata, float *idata, int width, int height) ;
__global__ void d_transpose_short(float *odata, short *idata, int width, int height);

void FFTonGPU(cufftComplex *GPUTimeDMArray,cufftComplex *GPUTimeDMArrayFFTbk, int x,int y, int index, cufftHandle plan);

void get_power_GPU(cufftComplex * data, int numdata, float *power);
__global__ void Do_get_power_GPU(cufftComplex * data, int numdata, float *power);


float  get_med_gpu(float *data, int N);


void spread_no_pad_gpu(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, float norm);
__global__ void Do_spread_with_pad_GPU(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, int numpad, float norm);

void spread_no_pad_gpu_dat(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, float norm, long long offset_bk, long long newnumbins_bk);
__global__ void Do_spread_with_pad_GPU_dat(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, int numpad, float norm, long long offset_bk, long long newnumbins_bk);

void spread_no_pad_gpu_list(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, int readdatanum, float *norm_data_gpu);
__global__ void Do_spread_with_pad_GPU_list(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, int numpad, int readdatanum, float *norm_data_gpu);


void spread_no_pad_gpu_dat_list_loiops(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, float *norm, long long offset_bk, long long numpad_bk, long long newnumbins_bk, long long addbins, int readdatanum, int data_len);
__global__ void Do_spread_with_pad_GPU_dat_list_loiops(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, int numpad, float *norm, long long offset_bk, long long numpad_bk, long long newnumbins_bk, long long addbins, int readdatanum, int data_len);

void loops_in_GPU_1(cufftComplex *fpdata, cufftComplex *fkern, cufftComplex *outdata, int fftlen, int numzs);
void loops_in_GPU_1_list(cufftComplex *fpdata, cufftComplex *fkern, cufftComplex *outdata, int fftlen, int numzs, int readdatanum);
static __global__ void Do_loops_in_GPU_1(cufftComplex *fpdata, cufftComplex *fkern, cufftComplex *outdata, int fftlen, int numzs);
static __global__ void Do_loops_in_GPU_1_list(cufftComplex *fpdata, cufftComplex *fkern, cufftComplex *outdata, int fftlen, int numzs, int readdatanum);


void loops_in_GPU_2(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip);
void loops_in_GPU_2_list(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip, int readdatanum, int outpows_gpu_xlen, long long outpows_gpu_obs_xlen, int *cand_num_gpu, accel_cand_gpu * cand_array_search_gpu, float powcut, int numharm, int output_x_max, int obs_fftlen);
static __global__ void Do_loops_in_GPU_2(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip);
static __global__ void Do_loops_in_GPU_2_list(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip, int readdatanum, int outpows_gpu_xlen, long long outpows_gpu_obs_xlen, int *d_addr, accel_cand_gpu * cand_array_search_gpu, float powcut, int numharm, int output_x_max, int obs_fftlen);
void loops_in_GPU_2_end(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip, accel_cand_gpu *cand_gpu_cpu, accel_cand_gpu * cand_array_search_gpu, int *nof_cand, float powcut, int numharm, int *cand_num_gpu);
static __global__ void Do_loops_in_GPU_2_end(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip, int *d_addr, accel_cand_gpu * cand_array_search_gpu, float powcut, int numharm);

void add_subharm_gpu(float *powers_out, cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int numzs,int offset, float norm);
void add_subharm_gpu_list(float *powers_out, cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int numzs,int offset, float norm, int readdatanum, int outpows_gpu_xlen, int obs_fftlen);
static __global__ void Do_add_subharm_gpu(float *powers_out, cufftComplex *fdata,unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int offset, float norm);
__global__ void Do_add_subharm_gpu_fast(
    float *powers_out,
    cufftComplex *fdata,
    unsigned short *rinds,
    unsigned short *zinds,
    int numrs,
    int numzs,
    int fftlen,
    int offset,
    float norm);

static __global__ void Do_add_subharm_gpu_list(float *powers_out, cufftComplex *fdata,unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int numzs, int offset, float norm, int readdatanum, int outpows_gpu_xlen, int obs_fftlen);

void add_subharm_gpu_norz(float *powers_out, cufftComplex *fdata, int numrs_0, int numzs_0, int fftlen, int numzs, int offset, float norm, int numharm, int harmnum, double fullrlo, double obszlo);
static __global__ void Do_add_subharm_gpu_norz(float *powers_out, cufftComplex *fdata,
    int numrs_0, int numzs_0, int fftlen, int offset, float norm,
    int numharm, int harmnum, double fullrlo, double obszlo);

void add_subharm_gpu_norz_1(float *powers_out, cufftComplex *fdata,
    int numrs_0, int numzs_0, int fftlen, int offset, float norm,
    double harm_fract, double drlo, long long rlo, double dzlo, long long zlo,
    double fullrlo, double obszlo);
static __global__ void Do_add_subharm_gpu_norz_1(float *powers_out, cufftComplex *fdata,
    int numrs_0, int numzs_0, int fftlen, int offset, float norm,
    double harm_fract, double drlo, long long rlo, double dzlo, long long zlo,
    double fullrlo, double obszlo);

static inline __device__ double calc_required_r_gpu(double harm_fract, double rfull);
static inline __device__ float calc_required_r_gpu_f(float harm_fract, float rfull);
static inline __device__ int index_from_r_gpu(double r, double lor);
static inline __device__ int index_from_r_gpu_f(float r, float lor);
static inline __device__ int calc_required_z_gpu(double harm_fract, double zfull);
static inline __device__ int calc_required_z_gpu_f(float harm_fract, float zfull);
static inline __device__ int index_from_z_gpu(double z, double loz);
static inline __device__ int index_from_z_gpu_f(float z, float loz);

void add_subharm_gpu_end(float *powers_out, cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int numzs, int offset, float norm, accel_cand_gpu *cand_gpu_cpu, accel_cand_gpu * cand_array_search_gpu, int *nof_cand, float powcut, int numharm, int *cand_num_gpu);
static __global__ void Do_add_subharm_gpu_end(float *powers_out, cufftComplex *fdata,unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int offset, float norm, int *d_addr, accel_cand_gpu * cand_array_search_gpu, float powcut, int numharm);
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
    int numharm);


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
    double obszlo);
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
    double obszlo);

void add_subharm_gpu_end_allharm(float *powers_out, cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, gpu_parameter_add *gpu_parameter_add_list, int numzs, accel_cand_gpu *cand_gpu_cpu, accel_cand_gpu * cand_array_search_gpu, int *nof_cand, float powcut, int numharm, int *cand_num_gpu, int fdata_len, int obs_corr_uselen, int obs_numz, int stage);
static __global__ void Do_add_subharm_gpu_end_allharm(float *powers_out, cufftComplex *fdata,unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, gpu_parameter_add *gpu_parameter_add_list, int *d_addr, accel_cand_gpu * cand_array_search_gpu, float powcut, int numharm, int fdata_len, int obs_corr_uselen, int obs_numz, int stage);


void add_subharm_gpu_cands_list(float *powers_out, cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int numzs, int offset, float norm, int readdatanum, int outpows_gpu_xlen, accel_cand_gpu *cand_gpu_cpu, accel_cand_gpu * cand_array_search_gpu, int *nof_cand, int output_x_max, float powcut, int numharm, int *cand_num_gpu, int obs_fftlen);
static __global__ void Do_add_subharm_gpu_cands_list(float *powers_out, cufftComplex *fdata,unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int numzs, int offset, float norm, int readdatanum, int outpows_gpu_xlen, int *d_addr, accel_cand_gpu * cand_array_search_gpu, float powcut, int output_x_max, int numharm, int obs_fftlen);

void add_subharm_gpu_cands_list_allharm(float *powers_out, cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, gpu_parameter_add *gpu_parameter_add_list, int numzs, int readdatanum, int outpows_gpu_xlen, accel_cand_gpu *cand_gpu_cpu, accel_cand_gpu * cand_array_search_gpu, int *nof_cand, int output_x_max, float powcut, int numharm, int *cand_num_gpu, int obs_fftlen, int fdata_len, int obs_corr_uselen, int obs_numz, int stage);
static __global__ void Do_add_subharm_gpu_cands_list_allharm(float *powers_out, cufftComplex *fdata,unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, gpu_parameter_add *gpu_parameter_add_list, int numzs, int readdatanum, int outpows_gpu_xlen, int *d_addr, accel_cand_gpu * cand_array_search_gpu, float powcut, int output_x_max, int numharm, int obs_fftlen, int fdata_len, int obs_corr_uselen, int obs_numz, int stage);

void inmem_add_ffdotpows_gpu_gpu(float *fdp, float *powptr, int *rinds, int zlo, int numrs, int numzs, int stage, long long rlen);
static __global__ void Do_inmem_add_ffdotpows_gpu_gpu(float *fdp, float *powptr, int *rinds,int zlo, int numrs, int numzs, int stage, long long rlen);



void inmem_add_subharm_and_search_ffdotpows_gpu(float *outpows_gpu_obs, float *outpows_gpu, int *rinds_gpu, int fd_zlo, int fd_numrs, int fd_numzs, int stage, long long obs_rlen, accel_cand_gpu * cand_array_search_gpu, accel_cand_gpu *cand_gpu_cpu, float powcut, int numharm, int *cand_num_gpu);
static  __global__ void Do_inmem_add_subharm_and_search_ffdotpows_gpu(float *outpows_gpu_obs, float *outpows_gpu, int *rinds,int zlo, int numrs, int numzs, int stage, long long rlen, float powcut, int *d_addr, accel_cand_gpu * cand_array_search_gpu, int numharm);

void inmem_add_ffdotpows_gpu_gpu_list(float *fdp, float *powptr, int *rinds, int zlo, int numrs, int numzs, int stage, long long rlen, int outpows_gpu_xlen, int readdatanum, int outpows_gpu_obs_xlen);
static  __global__ void Do_inmem_add_ffdotpows_gpu_gpu_list(float *fdp, float *powptr, int *rinds,int zlo, int numrs, int numzs, int stage, long long rlen, int outpows_gpu_xlen, int readdatanum, int outpows_gpu_obs_xlen);

void inmem_add_subharm_and_search_ffdotpows_gpu_list_m(float *outpows_gpu_obs, float *outpows_gpu, int *rinds_gpu, int fd_zlo, int fd_numrs, int fd_numzs, int stage, long long obs_rlen, accel_cand_gpu * cand_array_search_gpu, accel_cand_gpu *cand_gpu_cpu, float powcut, int output_x_max, int *nof_cand, long long outpows_gpu_obs_xlen, int readdatanum, int *cand_num_gpu, int harmtosum);
static  __global__ void Do_inmem_add_subharm_and_search_ffdotpows_gpu_list_m(float *outpows_gpu_obs, float *outpows_gpu, int *rinds,int zlo, int numrs, int numzs, int stage, long long rlen, float powcut, accel_cand_gpu * cand_array_search_gpu, int output_x_max, long long outpows_gpu_obs_xlen, int readdatanum, int *cand_num_gpu, int harmtosum);

int  search_ffdotpows_gpu(float powcut, float *d_fundamental, accel_cand_gpu * cand_array_search_gpu, int numzs, int numrs, accel_cand_gpu *cand_gpu_cpu);
void  search_ffdotpows_gpu_list_m(float powcut, float *d_fundamental, accel_cand_gpu * cand_array_search_gpu, int numzs, int numrs, accel_cand_gpu *cand_gpu_cpu, int readdatanum, int *nof_cand, int output_x_max, int d_fundamental_xlen);
static __global__ void  search_ffdotpows_kernel(float powcut, float *d_fundamental, accel_cand_gpu * cand_array_search_gpu, int numzs, int numrs, int *d_addr);
static __global__ void  search_ffdotpows_kernel_list_m(float powcut, float *d_fundamental, accel_cand_gpu * cand_array_search_gpu, int numzs, int numrs, int *d_addr, int readdatanum, int put_x_max, int d_fundamental_xlen);

void hunt_CPU(double *xx, int n, double x, int *jlo);
void hunt_GPU(double *xx, int n, double x, int *jlo);
double fold_gpu_cu(float *data, int nsub, int numdata, double dt, double tlo,
            double *prof, int numprof, double startphs,
            double *buffer, double *phaseadded,
            double fo, double fdot, double fdotdot, int flags,
            double *delays, double *delaytimes, int numdelays,
            int *onoffpairs, foldstats_gpu * stats, int standard, int ONOFF, int DELAYS, int worklen);


void fold_2_gpu(float *data, int nsub, int numdata, double dt, double tlo,
            double *prof, int numprof, double startphs,
            double *buffer, double *phaseadded,
            double fo, double fdot, double fdotdot, int flags,
            double *delays, double *delaytimes, int numdelays,
            int *onoffpairs, foldstats_gpu * stats, int standard, int worklen);


void Get_dmdelays_subband_int_gpu(double *search_dms_gpu, int *dmdelays, int numdmtrials, 
 int search_nsub,  double search_fold_p1, int search_proflen, double search_lofreq, 
 int search_numchan, double search_chan_wid, double search_avgvoverc);
__global__ void Get_dmdelays_subband_int_gpu_Do(double *search_dms_gpu, int *dmdelays, int numdmtrials, 
 int search_nsub,  double search_fold_p1, int search_proflen, double search_lofreq, 
 int search_numchan, double search_chan_wid, double search_avgvoverc);

void combine_subbands_1_gpu(double *inprofs,
                      int numparts, int numsubbands, int proflen, int numdmtrials,
                      int *delays, double *outprofs);
__global__ void combine_subbands_1_gpu_Do(double *inprofs,
                      int numparts, int numsubbands, int proflen, int numdmtrials,
                      int *delays, double *outprofs);

void get_delays_gpu(double *delays_gpu, const int numpdds, const int numpds, const int  numps, const int npart, const int numtrials, const int proflen, double *fdotdots, double *fdots, const int good_ipd, const int good_ip, const int searchpddP, const int search_pstep, const long reads_per_part, const double proftime);
__global__ void get_delays_gpu_Do(double *delays_gpu, const int numpdds, const int numpds, const int  numps, const int npart, const int numtrials, const int proflen, double *fdotdots, double *fdots, const int good_ipd, const int good_ip, const int searchpddP, const int search_pstep, const long reads_per_part, const double proftime);

void combine_profs_1_gpu(double *profs, double *delays, int numprofs, int proflen, int numpdd, int numpd, int nump, double outstats_prof_avg, double outstats_prof_var, double *currentstats_redchi);
__global__ void combine_profs_1_gpu_Do(double *profs, double *delays, int numprofs, int proflen, int numpdd, int numpd, int nump, double outstats_prof_avg, double outstats_prof_var, double *currentstats_redchi);



void combine_profs_2_gpu(double *ddprofs_gpu, double *search_pdots_gpu, double *search_periods_gpu, float *parttimes_gpu, int numpdots, int numperiods, int npart, int search_proflen, double outstats_prof_avg, double outstats_prof_var, float *currentstats_redchi, double pfold, double search_fold_p2, double search_fold_p1, double *pdd_delays_gpu, float chifact);
__global__ void combine_profs_2_gpu_Do(double *ddprofs_gpu, double *search_pdots_gpu, double *search_periods_gpu, float *parttimes_gpu, int numpdots, int numperiods, int npart, int search_proflen, double outstats_prof_avg, double outstats_prof_var, float *currentstats_redchi, double pfold, double search_fold_p2, double search_fold_p1, double *pdd_delays_gpu, float chifact);

__device__ double switch_pfdot_device(double pf, double pfdot);
__device__ double fdot2phasedelay_gpu(double fdot, double time);
__device__ float fdot2phasedelay_gpu_float(float fdot, float time);
__device__ double calculate_delays(double pf, double pfdot, double search_fold_p2, double df, double time);

void get_redchi_gpu(double *currentstats_redchi, double *outprof,  double outstats_prof_avg, double outstats_prof_var, int proflen, int parts);
__global__ void get_redchi_gpu_Do(double *currentstats_redchi, double *outprof,  double outstats_prof_avg, double outstats_prof_var, int proflen, int parts);
       

void Set_cufftComplex_date_as_zero_gpu(fcomplex *data, long long num);
__global__ void Set_cufftComplex_date_as_zero_gpu_Do(fcomplex *data, long long num);

void compute_power_gpu(fcomplex *data, float *powers, int numdata);
__global__ void compute_power_kernel(fcomplex *data, float *powers, int numdata);

void sort_and_get_median_gpu(float *data, int numdata, float *median);

/* Fused all-DM profile combination + per-DM argmax (prepfold_gpu) */
void combine_profs_all_dms_gpu(double *profs, double *delays, int numdms, int numprofs, int proflen, int numpdd, int numpd, int nump, double outstats_prof_avg, double outstats_prof_var, double *temp_chix, double *currentstats_redchi, int redchi_stride);
__global__ void combine_profs_all_dms_gpu_Do(double *profs, double *delays, long long numdms, int numprofs, int proflen, int numpdd, int numpd, int nump, double outstats_prof_avg, double outstats_prof_var, double *currentstats_redchi, int redchi_stride, int ntrial, long long total);
void get_maxindex_all_dms_gpu(double *redchi, int numdms, int stride, double *maxV, int *maxindex);
__global__ void get_maxindex_all_dms_gpu_Do(double *redchi, int numdms, int stride, double *maxV, int *maxindex);

/* Chunk-batched kernels used by accelsearch / accelsearchlist1 */
void spread_no_pad_gpu_dat_chunks(cufftComplex * data, int numdata,
                                  cufftComplex * result, int numresult,
                                  int numbetween, double fullrlo, double rstep,
                                  double harm_fract, int binoffset,
                                  long long obs_lobin, int readdatanum);
__global__ void Do_spread_with_pad_GPU_dat_chunks(const cufftComplex *data,
    int numdata, cufftComplex *result, int numresult, int numbetween,
    double fullrlo, double rstep, double harm_fract, int binoffset,
    long long obs_lobin, int readdatanum);

void add_subharm_gpu_chunkl_allharm(float *powers_out, cufftComplex *fdata,
    unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0,
    int K, int outpows_gpu_xlen, int output_x_max, float powcut, int numharm,
    int *cand_num_gpu, accel_cand_gpu * cand_array_search_gpu, int fdata_len,
    int obs_corr_uselen, int obs_numz, int stage_base, int nsub);
__global__ void Do_add_subharm_gpu_chunkl_allharm(float *powers_out,
    cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds,
    int numrs_0, int numzs_0, int K, int outpows_gpu_xlen, int output_x_max,
    float powcut, int numharm, int *d_addr,
    accel_cand_gpu * cand_array_search_gpu, int fdata_len,
    int obs_corr_uselen, int obs_numz, int stage, int nsub);


/* ---- 2026 batch optimization: GPU-side index & norm generation ---- */
/* Descriptor for one harm unit (fundamental + each (stage,harm) pair,
 * in call order).  Duplicated as plain bytes here; layout must match
 * the 32-byte aligned struct in cuda.cu. */
#ifdef __NVCC__
typedef struct __align__(32) mednorm_desc {
    double harm_fract;
    int    fftlen;
    int    binoffset;
    int    rsv0;
    int    rsv1;
} mednorm_desc;
#else
typedef struct __attribute__((aligned(32))) mednorm_desc {
    double harm_fract;
    int    fftlen;
    int    binoffset;
    int    rsv0;
    int    rsv1;
} mednorm_desc;
#endif

void mednorm_set_descs(const mednorm_desc *h, int n);
void rind_set_fracts(const double *h, int n);
void rind_gen_chunks_gpu(unsigned short *rinds, double fullrlo,
                         int corr_uselen, int nsub, int K);
void zind_gen_chunks_gpu(unsigned short *zinds, double obszlo, int numz,
                         int nsub, int K);
void mednorm_gen_gpu(const cufftComplex *fft, long long frame_stride,
                     long long base_pre, double fullrlo, double rstep,
                     int K, long long obs_lobin, long long obs_numbins,
                     int ncalls, float *norm_out);

void spread_no_pad_gpu_dat_chunks_gn(cufftComplex * data, int numdata,
                                     cufftComplex * result, int numresult,
                                     int numbetween, double fullrlo,
                                     double rstep, double harm_fract,
                                     int binoffset, long long obs_lobin,
                                     int readdatanum, const float *norms);
__global__ void Do_spread_with_pad_GPU_dat_chunks_gn(const cufftComplex *data,
    int numdata, cufftComplex *result, int numresult, int numbetween,
    double fullrlo, double rstep, double harm_fract, int binoffset,
    long long obs_lobin, int readdatanum, const float *norms);

void loops_in_GPU_1_list_f4(cufftComplex *fpdata, cufftComplex *fkern, cufftComplex *outdata, int fftlen, int numzs, int readdatanum);
__global__ void Do_loops_in_GPU_1_list_f4(const cufftComplex *fpdata,
    const cufftComplex *fkern, cufftComplex *outdata,
    int fftlen, int numzs, int readdatanum);
void add_subharm_gpu_cands_list_allharm_fast(float *powers_out, cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, gpu_parameter_add *gpu_parameter_add_list, int numzs, int readdatanum, int outpows_gpu_xlen, accel_cand_gpu *cand_gpu_cpu, accel_cand_gpu * cand_array_search_gpu, int *nof_cand, int output_x_max, float powcut, int numharm, int *cand_num_gpu, int obs_fftlen, int fdata_len, int obs_corr_uselen, int obs_numz, int stage);
void add_subharm_gpu_chunkl_allharm_fast(float *powers_out, cufftComplex *fdata,
    unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0,
    int K, int outpows_gpu_xlen, int output_x_max, float powcut, int numharm,
    int *cand_num_gpu, accel_cand_gpu * cand_array_search_gpu, int fdata_len,
    int obs_corr_uselen, int obs_numz, int stage_base, int nsub);

void spread_no_pad_gpu_dat_list_loiops_gn(cufftComplex * data, int numdata,
    cufftComplex *result, int numresult, int numbetween,
    long long offset_bk, long long numpad_bk, long long newnumbins_bk,
    long long addbins, int readdatanum, int data_len, const float *norms);
__global__ void Do_spread_with_pad_GPU_dat_list_loiops_gn(
    const cufftComplex *data, int numdata, cufftComplex *result,
    int numresult, int numbetween, long long offset_bk, long long addbins,
    int readdatanum, int data_len, const float *norms);

#ifdef __cplusplus
}
#endif
