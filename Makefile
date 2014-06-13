#gpu
CC = nvcc
OPT =--use_fast_math -O3
CURAND =-L/usr/local/cuda/lib64 -lcurand
#cpu
CCPU = gcc
OPTCPU =-ffast-math -O3

all: prog poisson dich double_prog double_poisson double_dich cpoisson dcpoisson cdich dcdich

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

cpoisson: poisson.c
	$(CCPU) $(OPTCPU) -o cpoisson poisson.c -lm
	cat cpoisson.py | sed 's/dpoisson/cpoisson/g' > dcpoisson.py

cdich: dich.c
	$(CCPU) $(OPTCPU) -o cdich dich.c -lm
	cat cdich.py | sed 's/ddich/cdich/g' > dcdich.py

dcpoisson: poisson.c
	cat poisson.c | sed 's/float/double/g;s/.0f/.0/g;s/.5f/.5/g;s/FLOAT/DOUBLE/g;s/%f/%lf/g;s/79f/79/g;s/fold(&/\/\/fold(&/g;s/\ +\ xfc//g' > dpoisson.c
	$(CCPU) -o dpoisson dpoisson.c -lm
	rm dpoisson.c
	cat cpoisson.py | sed 's/cpoisson/dpoisson/g' > dcpoisson.py

dcdich: dich.c cdich.py
	cat dich.c | sed 's/float/double/g;s/.0f/.0/g;s/.5f/.5/g;s/FLOAT/DOUBLE/g;s/%f/%lf/g;s/79f/79/g;s/fold(&/\/\/fold(&/g;s/\ +\ xfc//g' > ddich.c
	$(CCPU) -o ddich ddich.c -lm
	rm ddich.c
	cat cdich.py | sed 's/cdich/ddich/g' > dcdich.py
