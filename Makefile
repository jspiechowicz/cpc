OPT =--use_fast_math -O3
CURAND =-L/usr/local/cuda/lib64 -lcurand
CC = nvcc

all: prog poisson dich

prog: prog.cu
	$(CC) $(OPT) -o prog prog.cu $(CURAND) -lm

poisson: poisson.cu
	$(CC) $(OPT) -o poisson poisson.cu $(CURAND) -lm

dich: dich.cu
	$(CC) $(OPT) -o dich dich.cu $(CURAND) -lm
