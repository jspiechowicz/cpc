#gpu
CC = nvcc
OPT =--use_fast_math -O3
CURAND =-L/usr/local/cuda/lib64 -lcurand

#cpu
CCPU = gcc
OPTCPU =-ffast-math -O3

all: prog poisson dich dpoisson ddich cpoisson cdich dcpoisson dcdich

prog: prog.cu
	$(CC) $(OPT) -o prog prog.cu $(CURAND) -lm

poisson: poisson.cu
	$(CC) $(OPT) -o poisson poisson.cu $(CURAND) -lm

dich: dich.cu
	$(CC) $(OPT) -o dich dich.cu $(CURAND) -lm

dpoisson: poisson.cu poisson.py
	cat poisson.cu | sed 's/float/double/g;s/logf/log/g;s/expf/exp/g;s/sqrtf/sqrt/g;s/sinf/sin/g;s/cosf/cos/g;s/floorf/floor/g;s/curand_uniform/curand_uniform_double/g;s/curandGenerateUniform/curandGenerateUniformDouble/g;s/0f/0/g;s/5f/5/g' > dpoisson.cu
	$(CC) -o dpoisson dpoisson.cu $(CURAND) -lm
	rm dpoisson.cu
	cat poisson.py | sed 's/poisson/dpoisson/g' > dpoisson.py

ddich: dich.cu dich.py
	cat dich.cu | sed 's/float/double/g;s/logf/log/g;s/expf/exp/g;s/sqrtf/sqrt/g;s/sinf/sin/g;s/cosf/cos/g;s/floorf/floor/g;s/curand_uniform/curand_uniform_double/g;s/curandGenerateUniform/curandGenerateUniformDouble/g;s/0f/0/g;s/5f/5/g' > ddich.cu
	$(CC) -o ddich ddich.cu $(CURAND) -lm
	rm ddich.cu
	cat dich.py | sed 's/dich/ddich/g' > ddich.py

cpoisson: poisson.c
	$(CCPU) $(OPTCPU) -o cpoisson poisson.c -lm

cdich: dich.c
	$(CCPU) $(OPTCPU) -o cdich dich.c -lm

dcpoisson: poisson.c cpoisson.py
	cat poisson.c | sed 's/float/double/g;s/.0f/.0/g;s/.5f/.5/g;s/FLOAT/DOUBLE/g;s/%f/%lf/g;s/79f/79/g;s/fold(&/\/\/fold(&/g;s/\ +\ xfc//g' > dpoisson.c
	$(CCPU) -o dcpoisson dpoisson.c -lm
	rm dpoisson.c
	cat cpoisson.py | sed 's/cpoisson/dcpoisson/g' > dcpoisson.py

dcdich: dich.c cdich.py
	cat dich.c | sed 's/float/double/g;s/.0f/.0/g;s/.5f/.5/g;s/FLOAT/DOUBLE/g;s/%f/%lf/g;s/79f/79/g;s/fold(&/\/\/fold(&/g;s/\ +\ xfc//g' > ddich.c
	$(CCPU) -o dcdich ddich.c -lm
	rm ddich.c
	cat cdich.py | sed 's/cdich/dcdich/g' > dcdich.py

epl: prog.cu runepl.py epl.plt
	make prog
	python runepl.py

physletta: prog.cu runphysletta.py physletta.plt
	make prog
	python runphysletta.py

test: poisson.cu poisson.c poisson.py cpoisson.py interpolate.py test.plt
	make poisson
	make dpoisson
	make cpoisson
	make dcpoisson
	python poisson.py
	python dpoisson.py
	python cpoisson.py
	python dcpoisson.py
	python interpolate.py cpoisson_i7.dat
	python interpolate.py dcpoisson_i7.dat
	gnuplot test.plt

clean:
	rm -f prog
	rm -f poisson
	rm -f dpoisson
	rm -f dich
	rm -f ddich
	rm -f cpoisson
	rm -f dcpoisson
	rm -f cdich
	rm -f dcdich
	rm -f dpoisson.py
	rm -f ddich.py
	rm -f dcpoisson.py
	rm -f dcdich.py
