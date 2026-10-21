# CoherentSearch C build
#
# Uses the system/PSRSOFT FFTW3 (float and double).  Under the supplied bashrc
# (source /tempGPPS/CoherentSearch_C/bashrc) the include/library paths are in
# C_INCLUDE_PATH / LIBRARY_PATH already; the explicit -I/-L below also work.
#
#   make            # CPU-only binaries (GPU=0, the default)
#   make GPU=1      # include the CUDA/cuFFT backend
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

CFLAGS  ?= -O3 -march=native -ffp-contract=off -fopenmp -pthread -std=gnu11 -funroll-loops \
           -Wall -Wextra -Wno-unused-parameter -Wno-unused-function
CPPFLAGS += -I../include -I$(FFTW_INC)
LDFLAGS  += -fopenmp -pthread -L$(FFTW_LIB)
LDLIBS   += -lfftw3f -lfftw3 -lm

COMMON = fileio.c interp.c boxcar.c search.c candidate.c cli.c datpipe.c plot.c fftw_lock.c zap.c
COMMON_OBJ = $(COMMON:.c=.o)

ifeq ($(GPU),1)
CPPFLAGS += -DCS_HAVE_CUDA
GPU_OBJ = cuda.o
CUDA_ARCH ?= -arch=native
NVCCFLAGS ?= -O3 $(CUDA_ARCH) -std=c++14 -I../include -I$(FFTW_INC) -I$(CUDA_DIR)/include
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
