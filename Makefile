CFLAGS =-arch=sm_20 -m64 --use_fast_math -O3
CURAND =-L/usr/local/cuda/lib64 -lcurand
CC = nvcc

all: prog

prog: prog.cu
	$(CC) $(CFLAGS) -o prog prog.cu $(CURAND) -lm
