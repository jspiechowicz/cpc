/*
 * Overdamped Brownian particle in symmetric piecewise linear potential
 *
 * \dot{x} = -V'(x) + Poissonian noise
 *
 */

#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define PI 3.14159265358979f

//model
__constant__ float d_Dp, d_lambda, d_mean;
float h_lambda, h_mean;

//simulation
float h_trans;
int h_dev, h_block, h_grid, h_spp, h_samples;
long h_paths, h_periods, h_threads, h_steps, h_trigger;
__constant__ int d_spp, d_samples;
__constant__ long d_paths;

//output
char *h_domain;
char h_domainx;
float h_beginx, h_endx;
int h_logx, h_points, h_moments;
__constant__ char d_domainx;
__constant__ int d_points;

//vector
float *h_x, *h_xb, *h_fx, *h_dx;
float *d_x, *d_fx, *d_dx;
int *d_pcd;
unsigned int *h_seeds, *d_seeds;
curandState *d_states;

size_t size_f, size_i, size_ui, size_p;
curandGenerator_t gen;

static struct option options[] = {
    {"Dp", required_argument, NULL, 'a'},
    {"lambda", required_argument, NULL, 'b'},
    {"mean", required_argument, NULL, 'c'},
    {"dev", required_argument, NULL, 'd'},
    {"block", required_argument, NULL, 'e'},
    {"paths", required_argument, NULL, 'f'},
    {"periods", required_argument, NULL, 'g'},
    {"trans", required_argument, NULL, 'h'},
    {"spp", required_argument, NULL, 'i'},
    {"samples", required_argument, NULL, 'j'},
    {"mode", required_argument, NULL, 'k'},
    {"domain", required_argument, NULL, 'l'},
    {"domainx", required_argument, NULL, 'm'},
    {"logx", required_argument, NULL, 'n'},
    {"points", required_argument, NULL, 'o'},
    {"beginx", required_argument, NULL, 'p'},
    {"endx", required_argument, NULL, 'q'}
};

void usage(char **argv)
{
    printf("Usage: %s <params> \n\n", argv[0]);
    printf("Model params:\n");
    printf("    -a, --Dp=FLOAT          set the Poissonian noise intensity 'D_P' to FLOAT\n");
    printf("    -b, --lambda=FLOAT      set the Poissonian kicks frequency '\\lambda' to FLOAT\n");
    printf("    -c, --mean=FLOAT        if is nonzero, fix the mean value of Poissonian noise to FLOAT, matters only for domains p, l\n");
    printf("Simulation params:\n");
    printf("    -d, --dev=INT           set the gpu device to INT\n");
    printf("    -e, --block=INT         set the gpu block size to INT\n");
    printf("    -f, --paths=LONG        set the number of paths to LONG\n");
    printf("    -g, --periods=LONG      set the number of periods to LONG\n");
    printf("    -h, --trans=FLOAT       specify fraction FLOAT of periods which stands for transients\n");
    printf("    -i, --spp=INT           specify how many integration steps should be calculated for a characteristic time scale of Poissonian noise\n");
    printf("    -j, --samples=INT       specify how many integration steps should be calculated for a single kernel call\n");
    printf("Output params:\n");
    printf("    -k, --mode=STRING       sets the output mode. STRING can be one of:\n");
    printf("                            moments: the first moment <<v>>\n");
    printf("    -l, --domain=STRING     simultaneously scan over one or two model params. STRING can be one of:\n");
    printf("                            1d: only one parameter\n");
    printf("    -m, --domainx=CHAR      sets the first domain of the moments. CHAR can be one of:\n");
    printf("                            p: Dp; l: lambda\n");
    printf("    -n, --logx=INT          choose between linear and logarithmic scale of the domainx\n");
    printf("                            0: linear; 1: logarithmic\n");
    printf("    -o, --points=INT        set the number of samples to generate between begin and end\n");
    printf("    -p, --beginx=FLOAT      set the starting value of the domainx to FLOAT\n");
    printf("    -q, --endx=FLOAT        set the end value of the domainx to FLOAT\n");
    printf("\n");
}

void parse_cla(int argc, char **argv)
{
    float ftmp;
    int c, itmp;

    while( (c = getopt_long(argc, argv, "a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q", options, NULL)) != EOF) {
        switch (c) {
            case 'a':
                ftmp = atof(optarg);
                cudaMemcpyToSymbol(d_Dp, &ftmp, sizeof(float));
                break;
            case 'b':
                h_lambda = atof(optarg);
                cudaMemcpyToSymbol(d_lambda, &h_lambda, sizeof(float));
                break;
            case 'c':
                h_mean = atof(optarg);
                cudaMemcpyToSymbol(d_mean, &h_mean, sizeof(float));
                break;
            case 'd':
                itmp = atoi(optarg);
                cudaSetDevice(itmp);
                break;
            case 'e':
                h_block = atoi(optarg);
                break;
            case 'f':
                h_paths = atol(optarg);
                cudaMemcpyToSymbol(d_paths, &h_paths, sizeof(long));
                break;
            case 'g':
                h_periods = atol(optarg);
                break;
            case 'h':
                h_trans = atof(optarg);
                break;
            case 'i':
                h_spp = atoi(optarg);
                cudaMemcpyToSymbol(d_spp, &h_spp, sizeof(int));
                break;
            case 'j':
                h_samples = atoi(optarg);
                cudaMemcpyToSymbol(d_samples, &h_samples, sizeof(int));
                break;
            case 'k':
                if ( !strcmp(optarg, "moments") ) {
                    h_moments = 1;
                }
                break;
            case 'l':
                h_domain = optarg;
                break;
            case 'm':
                h_domainx = optarg[0]; 
                cudaMemcpyToSymbol(d_domainx, &h_domainx, sizeof(char));
                break;
            case 'n':
                h_logx = atoi(optarg);
                break;
            case 'o':
                h_points = atoi(optarg);
                cudaMemcpyToSymbol(d_points, &h_points, sizeof(int));
                break;
            case 'p':
                h_beginx = atof(optarg);
                break;
            case 'q':
                h_endx = atof(optarg);
                break;
        }
    }
}

__global__ void init_dev_rng(unsigned int *d_seeds, curandState *d_states)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(d_seeds[idx], 0, 0, &d_states[idx]);
}

__device__ float drift(float l_x)
{    
    if (-sinf(PI*l_x) < 0.0f) {
        return -1.0f;
    } else {
        return 1.0f;
    }
}

__global__ void init_poisson(float *d_dx, int *d_pcd, curandState *d_states)
//init Poissonian noise
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    float l_dx;
    curandState l_state;

    //cache model parameters in local variables
    l_state = d_states[idx];
    
    float l_Dp, l_lambda, l_mean;
    
    l_Dp = d_Dp;
    l_lambda = d_lambda;
    l_mean = d_mean;

    long ridx = (idx/d_paths) % d_points;
    l_dx = d_dx[ridx];

    switch(d_domainx) {
        case 'p':
            l_Dp = l_dx;
            if (l_mean != 0.0f) l_lambda = (l_mean*l_mean)/l_Dp;
            break;
        case 'l':
            l_lambda = l_dx;
            if (l_mean != 0.0f) l_Dp = (l_mean*l_mean)/l_lambda;
            break;
    }

    //step size
    float l_dt;
    int l_spp;

    l_spp = d_spp;
    l_dt = 1.0f/l_lambda/l_spp; 

    //jump countdown
    int l_pcd;
    
    l_pcd = (int) floorf( -logf( curand_uniform(&l_state) )/l_lambda/l_dt + 0.5f );
    
    //write back noise state to the global memory
    d_pcd[idx] = l_pcd;
    d_states[idx] = l_state;
}

__device__ float adapted_jump_poisson(int &npcd, int pcd, float l_lambda, float l_Dp, float l_dt, curandState *l_state)
{
    float comp = sqrtf(l_Dp*l_lambda)*l_dt;
    
    if (pcd <= 0) {
        float ampmean = sqrtf(l_lambda/l_Dp);
        
        npcd = (int) floorf( -logf( curand_uniform(l_state) )/l_lambda/l_dt + 0.5f );
    
        return -logf( curand_uniform(l_state) )/ampmean - comp;
    } else {
        npcd = pcd - 1;

        return -comp;
    }
}

__device__ void predcorr(float &corrl_x, float l_x, int &npcd, int pcd, curandState *l_state, \
                         float l_Dp, float l_lambda, float l_dt)
/* simplified weak order 2.0 adapted predictor-corrector scheme
( see E. Platen, N. Bruti-Liberati; Numerical Solution of Stochastic Differential Equations with Jumps in Finance; Springer 2010; p. 503, p. 532 )
*/
{
    float l_xt, l_xtt, predl_x;

    l_xt = drift(l_x);

    predl_x = l_x + l_xt*l_dt; 

    l_xtt = drift(predl_x);

    predl_x = l_x + 0.5f*(l_xt + l_xtt)*l_dt;

    l_xtt = drift(predl_x);

    corrl_x = l_x + 0.5f*(l_xt + l_xtt)*l_dt + adapted_jump_poisson(npcd, pcd, l_lambda, l_Dp, l_dt, l_state);
}

__global__ void fold(float *d_x, float *d_fx)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    float l_x, l_fx, f;

    l_x = d_x[idx];
    l_fx = d_fx[idx];

    f = floorf(l_x/2.0f)*2.0f;
    l_x = l_x - f;
    l_fx = l_fx + f;

    d_x[idx] = l_x;
    d_fx[idx] = l_fx;
}

void unfold(float *x, float *fx)
{
    int i;

    for (i = 0; i < h_threads; i++) {
        x[i] = x[i] + fx[i];
    }
}

__global__ void run_moments(float *d_x, float *d_dx, int *d_pcd, curandState *d_states)
//actual moments kernel
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    float l_x, l_dx; 
    int l_pcd;
    curandState l_state;

    //cache path and model parameters in local variables
    l_x = d_x[idx];
    l_pcd = d_pcd[idx];
    l_state = d_states[idx];

    float l_Dp, l_lambda, l_mean;

    l_Dp = d_Dp;
    l_lambda = d_lambda;
    l_mean = d_mean;

    //run simulation for multiple values of the system parameters
    long ridx = (idx/d_paths) % d_points;
    l_dx = d_dx[ridx];

    switch(d_domainx) {
        case 'p':
            l_Dp = l_dx;
            if (l_mean != 0.0f) l_lambda = (l_mean*l_mean)/l_Dp;
            break;
        case 'l':
            l_lambda = l_dx;
            if (l_mean != 0.0f) l_Dp = (l_mean*l_mean)/l_lambda;
            break;
    }

    //step size & number of steps
    float l_dt;
    int i, l_spp, l_samples;

    l_spp = d_spp;
    l_dt = 1.0f/l_lambda/l_spp; 
    l_samples = d_samples;

    for (i = 0; i < l_samples; i++) {
        predcorr(l_x, l_x, l_pcd, l_pcd, &l_state, l_Dp, l_lambda, l_dt);
    }

    //write back path parameters to the global memory
    d_x[idx] = l_x;
    d_pcd[idx] = l_pcd;
    d_states[idx] = l_state;
}

void prepare()
//prepare simulation
{
    //grid size
    h_paths = (h_paths/h_block)*h_block;
    h_threads = h_paths;

    if (h_moments) h_threads *= h_points;

    h_grid = h_threads/h_block;

    //number of steps
    if (h_moments) h_steps = h_periods*h_spp;
     
    //host memory allocation
    size_f = h_threads*sizeof(float);
    size_i = h_threads*sizeof(int);
    size_ui = h_threads*sizeof(unsigned int);
    size_p = h_points*sizeof(float);

    h_x = (float*)malloc(size_f);
    h_seeds = (unsigned int*)malloc(size_ui);

    //create & initialize host rng
    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

    curandGenerate(gen, h_seeds, h_threads);
 
    //device memory allocation
    cudaMalloc((void**)&d_x, size_f);
    cudaMalloc((void**)&d_seeds, size_ui);
    cudaMalloc((void**)&d_pcd, size_i);
    cudaMalloc((void**)&d_states, h_threads*sizeof(curandState));

    //copy seeds from host to device
    cudaMemcpy(d_seeds, h_seeds, size_ui, cudaMemcpyHostToDevice);

    //initialization of device rng
    init_dev_rng<<<h_grid, h_block>>>(d_seeds, d_states);

    free(h_seeds);
    cudaFree(d_seeds);

    //moments specific requirements
    if (h_moments) {
        h_trigger = h_steps*h_trans;

        h_xb = (float*)malloc(size_f);
        h_fx = (float*)malloc(size_f);
        h_dx = (float*)malloc(size_p);

        float dxtmp = h_beginx;
        float dxstep = (h_endx - h_beginx)/h_points;

        int i;
        
        //set domainx
        for (i = 0; i < h_points; i++) {
            if (h_logx) {
                h_dx[i] = exp10f(dxtmp);
            } else {
                h_dx[i] = dxtmp;
            }
            dxtmp += dxstep;
        }
        
        cudaMalloc((void**)&d_fx, size_f);
        cudaMalloc((void**)&d_dx, size_p);
    
        cudaMemcpy(d_dx, h_dx, size_p, cudaMemcpyHostToDevice);
    }
}

void copy_to_dev()
{
    cudaMemcpy(d_x, h_x, size_f, cudaMemcpyHostToDevice);
}

void copy_from_dev()
{
    cudaMemcpy(h_x, d_x, size_f, cudaMemcpyDeviceToHost);
}

void initial_conditions()
//set initial conditions for path parameters
{
    int i;

    curandGenerateUniform(gen, h_x, h_threads); //x in (0,1]

    for (i = 0; i < h_threads; i++) {
        h_x[i] = 2.0f*h_x[i] - 1.0f; //x in (-1,1]
    }

    copy_to_dev();
}

void moments(float *av)
//calculate the first moment of v
{
    float sx, sxb, dt;
    int i, j;

    cudaMemcpy(h_x, d_x, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fx, d_fx, size_f, cudaMemcpyDeviceToHost);

    unfold(h_x, h_fx);

    for (j = 0; j < h_points; j++) {
        sx = 0.0f;
        sxb = 0.0f;

        for (i = 0; i < h_paths; i++) {
            sx += h_x[j*h_paths + i];
            sxb += h_xb[j*h_paths + i];
        }

        //Poissonian
        if (h_domainx == 'l') {
            dt = 1.0f/h_dx[j]/h_spp;
        } else if (h_domainx == 'p' && h_mean != 0.0f) {
            dt = 1.0f/(h_mean*h_mean/h_dx[j])/h_spp;
        } else if (h_lambda != 0.0f) {
            dt = 1.0f/h_lambda/h_spp;
        }

        av[j] = (sx - sxb)/( (1.0f - h_trans)*h_steps*dt )/h_paths;
    }
}

void finish()
//free memory
{

    free(h_x);
    
    curandDestroyGenerator(gen);
    cudaFree(d_x);
    cudaFree(d_pcd);
    cudaFree(d_states);
    
    if (h_moments) {
        free(h_xb);
        free(h_fx);
        free(h_dx);

        cudaFree(d_fx);
        cudaFree(d_dx);
    }
}

int main(int argc, char **argv)
{
    clock_t b, e;
    double t;

    parse_cla(argc, argv);
    if (!h_moments) {
        usage(argv);
        return -1;
    }
 
    prepare();
    
    initial_conditions();
    
    //asymptotic long time average velocity <<v>>
    if (h_moments) {
        //float *av;
        int i;

        //av = (float*)malloc(size_p);
 
        if ( !strcmp(h_domain, "1d") ) { 

            cudaDeviceSynchronize();
            b = clock();

            init_poisson<<<h_grid, h_block>>>(d_dx, d_pcd, d_states);

            for (i = 0; i < h_steps; i += h_samples) {
                run_moments<<<h_grid, h_block>>>(d_x, d_dx, d_pcd, d_states);
                fold<<<h_grid, h_block>>>(d_x, d_fx);
                if (i == h_trigger) {
                    cudaMemcpy(h_xb, d_x, size_f, cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_fx, d_fx, size_f, cudaMemcpyDeviceToHost);
                    unfold(h_xb, h_fx);
                }
            }

            cudaDeviceSynchronize();
            e = clock();
            t = (double)(e - b) / CLOCKS_PER_SEC;

            /*moments(av);
 
            printf("#%c <<v>>\n", h_domainx);
            for (i = 0; i < h_points; i++) {
                printf("%e %e\n", h_dx[i], av[i]);
            }*/

            printf("%lf\n", t);
        }

        //free(av);
    }

    finish();
    
    return 0;
}
