# CoherentSearch C build
#
# Uses the system/PSRSOFT FFTW3 (float and double).  Under the supplied bashrc
# (source /tempGPPS/CoherentSearch_C/bashrc) the include/library paths are in
# C_INCLUDE_PATH / LIBRARY_PATH already; the explicit -I/-L below also work.
#
#   make            # CPU-only binaries (GPU=0, the default)
#   make GPU=1      # include the CUDA/cuFFT backend
#   make LTO=1      # add link-time optimization (GCC; same results)
#   make VECTOR_WIDTH=256   # or 128/512/auto; see below
#
# Two main programs are built:
#   bin/coherent_search       FILE.fft|FILE.dat [more inputs] [options]
#   bin/coherent_search_list  FILE.dat.list [options]
#
# A raw-data front end is also built:
#   bin/prep_coherent_search       <prepsubband opts> ++ <search opts>
# It dedisperses raw PSRFITS/filterbank/cache data in memory, writes the .inf
# files, and searches without creating any .dat file.  GPU=0 builds it with
# CPU-only dedispersion and no CUDA linkage; GPU=1 also enables -cuda
# dedispersion and the GPU search backend.

.DEFAULT_GOAL := all

CC       = gcc
NVCC     = nvcc
FFTW_INC ?= /home/gpps/pulsar_software/include
FFTW_LIB ?= /home/gpps/pulsar_software/lib
CUDA_DIR ?= /usr/local/cuda
PRESTO_VENDOR ?= ../third_party/presto
PRESTO_SRC ?= $(PRESTO_VENDOR)/src
PRESTO_INC ?= $(PRESTO_VENDOR)/include
GPU      ?= 0
LTO      ?= 0
# Vector width preference:
#   native (default) -> let -march=native decide;
#   auto             -> 256 on AMD AVX-512, 512 on Intel AVX-512, else native;
#   128/256/512      -> force that preferred width.
# This only changes GCC's vector-width preference; it does not change the ISA
# selected by -march=native or the numerical results.
VECTOR_WIDTH ?= native

CFLAGS  ?= -O3 -march=native -ffp-contract=off -fopenmp -pthread -std=gnu11 -funroll-loops \
           -Wall -Wextra -Wno-unused-parameter -Wno-unused-function
CPPFLAGS += -I../include -I$(FFTW_INC)
LDFLAGS  += -fopenmp -pthread -L$(FFTW_LIB)
LDLIBS   += -lfftw3f -lfftw3 -lm

VW := $(VECTOR_WIDTH)
ifeq ($(VW),auto)
  CPU_VENDOR := $(shell sed -n 's/^vendor_id[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo 2>/dev/null | head -n1)
  CPU_AVX512 := $(shell grep -m1 -o avx512f /proc/cpuinfo 2>/dev/null)
  ifeq ($(CPU_AVX512),avx512f)
    ifeq ($(CPU_VENDOR),GenuineIntel)
      VW := 512
    else
      VW := 256
    endif
  else
    VW := native
  endif
endif
ifeq ($(filter $(VW),native none 128 256 512),)
  $(error VECTOR_WIDTH must be native, auto, none, 128, 256, or 512)
endif
ifneq ($(VW),native)
ifneq ($(VW),none)
CFLAGS += -mprefer-vector-width=$(VW)
endif
endif

# Link-time optimization: interprocedural inlining/constant propagation across
# the C translation units.  It does not relax FP contraction or reassociation,
# so results are unchanged.  Measured gain is small (~1-3% on the 105-file CPU
# list) but the switch is free when the toolchain supports it.
ifeq ($(LTO),1)
CFLAGS  += -flto=auto -fno-semantic-interposition
LDFLAGS += -flto=auto
endif

COMMON = fileio.c interp.c boxcar.c search.c candidate.c cli.c datpipe.c plot.c fftw_lock.c zap.c
COMMON_OBJ = $(COMMON:.c=.o)

ifeq ($(GPU),1)
CPPFLAGS += -DCS_HAVE_CUDA
GPU_OBJ = cuda.o
PREP_CUDA_OBJ = presto_cuda.o
CUDA_ARCH ?= -arch=native
NVCC_EXTRA ?=
NVCCFLAGS ?= -O3 $(CUDA_ARCH) -std=c++14 -I../include -I$(FFTW_INC) -I$(CUDA_DIR)/include $(NVCC_EXTRA)
LDFLAGS += -L$(CUDA_DIR)/lib64 -Wl,-rpath,$(CUDA_DIR)/lib64
LDLIBS  += -lcufft -lcudart
else
GPU_OBJ = gpu_stub.o
PREP_CUDA_OBJ = presto_gpu_cpu.o
endif

BIN     = ../bin/coherent_search
BINLIST = ../bin/coherent_search_list

# Raw-data dedispersion is compiled from the vendored PRESTO sources under
# third_party/presto (no external PRESTO/libpresto).  cfitsio/CUDA are external,
# so these two programs are built in GPU=1
# configurations (the search itself may still run on the CPU with no --gpu).
PREP_C_SRCS = backend_common psrfits sigproc_fb zerodm mask prepsubband_cmd \
              ioinf misc_utils chkio barycenter clipping dispersion vectors \
              range_parse transpose cldj djcl sorter median birdzap
PREP_OBJS = $(addprefix presto_,$(addsuffix .o,$(PREP_C_SRCS))) $(PREP_CUDA_OBJ)
PREP_BINS = ../bin/prep_coherent_search

all: $(BIN) $(BINLIST) $(PREP_BINS)

$(BIN): CoherentSearch.o $(COMMON_OBJ) $(GPU_OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BINLIST): list_main.o $(COMMON_OBJ) $(GPU_OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

PREP_COMMON = prep_search.o prep_cli.o prep_dedisp.o $(PREP_OBJS)
PREP_LDFLAGS = $(LDFLAGS)
PREP_LDLIBS = -lcfitsio -lglib-2.0 -lstdc++ -lmvec

../bin/prep_coherent_search: prep_main.o $(PREP_COMMON) $(COMMON_OBJ) $(GPU_OBJ)
	$(CC) $(PREP_LDFLAGS) -o $@ $^ $(LDLIBS) $(PREP_LDLIBS)

%.o: %.c ../include/coherent_search.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

# PRESTO's realfft/rednoise were compiled with default FP contraction; matching
# it here reproduces their last-bit arithmetic most closely.
datpipe.o: datpipe.c ../include/coherent_search.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -ffp-contract=fast -c -o $@ $<

# PRESTO's zapbirds was built with default FP contraction; match its last-bit
# medians/amplitudes.
zap.o: zap.c ../include/coherent_search.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -ffp-contract=fast -c -o $@ $<

cuda.o: cuda.cu ../include/coherent_search.h ../include/cuda.cuh
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

prep_search.o: prep_search.c ../include/coherent_search.h ../include/prep_search.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

prep_cli.o: prep_cli.c ../include/coherent_search.h ../include/prep_search.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

prep_main.o: prep_main.c ../include/prep_search.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

# PRESTO headers must come first: prepsubband includes its own cuda.cuh.
# PRESTO's prepsubband was built with -Ofast.  Match that arithmetic for the
# borrowed raw-data/dedispersion sources; the search kernels keep strict FP.
PREP_CFLAGS = $(CFLAGS) -Ofast
prep_dedisp.o: prep_dedisp.c ../include/prep_search.h
	$(CC) -I$(PRESTO_INC) $(CPPFLAGS) $(PREP_CFLAGS) -fcommon -c -o $@ $<

# PRESTO's headers contain tentative globals; build these sources with
# -fcommon so they merge with the ones inside prep_dedisp.o.
presto_%.o: $(PRESTO_SRC)/%.c
	$(CC) -I$(PRESTO_INC) $(CPPFLAGS) $(PREP_CFLAGS) -fcommon -c -o $@ $<

PREP_NVCCFLAGS = -O3 $(CUDA_ARCH) -std=c++14 -I$(PRESTO_INC) -I../include \
                 -I$(FFTW_INC) -I$(CUDA_DIR)/include $(NVCC_EXTRA)
presto_cuda.o: $(PRESTO_SRC)/cuda.cu $(PRESTO_INC)/cuda.cuh
	$(NVCC) $(PREP_NVCCFLAGS) -c -o $@ $<

presto_gpu_cpu.o: presto_gpu_cpu.c $(PRESTO_INC)/presto_gpu_cpu.h
	$(CC) -I$(PRESTO_INC) $(CPPFLAGS) $(PREP_CFLAGS) -c -o $@ $<

clean:
	rm -f $(COMMON_OBJ) CoherentSearch.o list_main.o \
	      cuda.o gpu_stub.o \
	      prep_search.o prep_cli.o prep_dedisp.o prep_main.o \
	      $(addprefix presto_,$(addsuffix .o,$(PREP_C_SRCS))) presto_cuda.o \
	      presto_gpu_cpu.o # $(BINLIST) $(BIN)

.PHONY: all clean
