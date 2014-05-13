OPT =--use_fast_math -O3
CURAND =-L/usr/local/cuda/lib64 -lcurand
CC = nvcc

all: single double

single: prog.cu
	$(CC) $(OPT) -o prog prog.cu $(CURAND) -lm

double: double_prog.cu
	$(CC) -o prog double_prog.cu $(CURAND) -lm
