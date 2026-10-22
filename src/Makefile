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
# Two programs are built:
#   bin/coherent_search       FILE.fft|FILE.dat [more inputs] [options]
#   bin/coherent_search_list  FILE.dat.list [options]
#
# On a platform without CUDA, keep GPU=0; the binaries still accept --gpu but
# warn and run the CPU path.

CC       = gcc
NVCC     = nvcc
FFTW_INC ?= /home/gpps/pulsar_software/include
FFTW_LIB ?= /home/gpps/pulsar_software/lib
CUDA_DIR ?= /usr/local/cuda
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
CUDA_ARCH ?= -arch=native
NVCC_EXTRA ?=
NVCCFLAGS ?= -O3 $(CUDA_ARCH) -std=c++14 -I../include -I$(FFTW_INC) -I$(CUDA_DIR)/include $(NVCC_EXTRA)
LDFLAGS += -L$(CUDA_DIR)/lib64 -Wl,-rpath,$(CUDA_DIR)/lib64
LDLIBS  += -lcufft -lcudart
else
GPU_OBJ = gpu_stub.o
endif

BIN     = ../bin/coherent_search
BINLIST = ../bin/coherent_search_list

all: $(BIN) $(BINLIST)

$(BIN): CoherentSearch.o $(COMMON_OBJ) $(GPU_OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BINLIST): list_main.o $(COMMON_OBJ) $(GPU_OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

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

clean:
	rm -f $(COMMON_OBJ) CoherentSearch.o list_main.o cuda.o gpu_stub.o # $(BINLIST) $(BIN)

.PHONY: all clean
