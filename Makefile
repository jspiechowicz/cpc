OPT =--use_fast_math -O3
CURAND =-L/usr/local/cuda/lib64 -lcurand
CC = nvcc

all: prog poisson dich double_prog double_poisson double_dich

single: prog poisson dich

double: double_prog double_poisson double_dich

prog: prog.cu
	$(CC) $(OPT) -o prog prog.cu $(CURAND) -lm

poisson: poisson.cu
	$(CC) $(OPT) -o poisson poisson.cu $(CURAND) -lm

dich: dich.cu
	$(CC) $(OPT) -o dich dich.cu $(CURAND) -lm

double_prog: double_prog.cu
	$(CC) -o double_prog double_prog.cu $(CURAND) -lm

double_poisson: double_poisson.cu
	$(CC) -o double_poisson double_poisson.cu $(CURAND) -lm

double_dich: double_dich.cu
	$(CC) -o double_dich double_dich.cu $(CURAND) -lm
