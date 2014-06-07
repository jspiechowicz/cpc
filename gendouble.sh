#!/bin/bash 

progs='prog poisson dich'

for i in $progs; do
    sed -e 's/float/double/g;
            s/logf/log/g; 
            s/expf/exp/g;
            s/sqrtf/sqrt/g;
            s/sinf/sin/g; 
            s/cosf/cos/g;
            s/floorf/floor/g;
            s/curand_uniform/curand_uniform_double/g;
            s/curandGenerateUniform/curandGenerateUniformDouble/g;
            s/0f/0/g;
            s/5f/5/g' ${i}.cu > double_${i}.cu
done
