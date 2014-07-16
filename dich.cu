/*
 * Overdamped Brownian particle in symmetric piecewise linear potential
 *
 * \dot{x} = -V'(x) + dichotomous noise
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
__constant__ float d_fa, d_fb, d_mua, d_mub;
__constant__ int d_comp;
float h_fa, h_fb, h_mua, h_mub;
int h_comp;

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
int *d_dcd, *d_dst;
unsigned int *h_seeds, *d_seeds;
curandState *d_states;

size_t size_f, size_i, size_ui, size_p;
curandGenerator_t gen;

static struct option options[] = {
    {"fa", required_argument, NULL, 'a'},
    {"fb", required_argument, NULL, 'b'},
    {"mua", required_argument, NULL, 'c'},
    {"mub", required_argument, NULL, 'd'},
    {"comp", required_argument, NULL, 'e'},
    {"dev", required_argument, NULL, 'f'},
    {"block", required_argument, NULL, 'g'},
    {"paths", required_argument, NULL, 'h'},
    {"periods", required_argument, NULL, 'i'},
    {"trans", required_argument, NULL, 'j'},
    {"spp", required_argument, NULL, 'k'},
    {"samples", required_argument, NULL, 'l'},
    {"mode", required_argument, NULL, 'm'},
    {"domain", required_argument, NULL, 'n'},
    {"domainx", required_argument, NULL, 'o'},
    {"logx", required_argument, NULL, 'p'},
    {"points", required_argument, NULL, 'q'},
    {"beginx", required_argument, NULL, 'r'},
    {"endx", required_argument, NULL, 's'}
};

void usage(char **argv)
{
    printf("Usage: %s <params> \n\n", argv[0]);
    printf("Model params:\n");
    printf("    -a, --fa=FLOAT          set the first state of the dichotomous noise 'F_a' to FLOAT\n");
    printf("    -b, --fb=FLOAT          set the second state of the dichotomous noise 'F_b' to FLOAT\n");
    printf("    -c, --mua=FLOAT         set the transition rate of the first state of dichotomous noise '\\mu_a' to FLOAT\n");
    printf("    -d, --mub=FLOAT         set the transition rate of the second state of dichotomous noise '\\mu_b' to FLOAT\n");
    printf("    -e, --comp=INT          choose between biased and unbiased Poissonian or dichotomous noise. INT can be one of:\n");
    printf("                            0: biased; 1: unbiased\n");
    printf("Simulation params:\n");
    printf("    -f, --dev=INT           set the gpu device to INT\n");
    printf("    -g, --block=INT         set the gpu block size to INT\n");
    printf("    -h, --paths=LONG        set the number of paths to LONG\n");
    printf("    -i, --periods=LONG      set the number of periods to LONG\n");
    printf("    -j, --trans=FLOAT       specify fraction FLOAT of periods which stands for transients\n");
    printf("    -k, --spp=INT           specify how many integration steps should be calculated for the smaller characteristic time scale of dichotomous noise\n");
    printf("    -l, --samples=INT       specify how many integration steps should be calculated for a single kernel call\n");
    printf("Output params:\n");
    printf("    -m, --mode=STRING       sets the output mode. STRING can be one of:\n");
    printf("                            moments: the first moment <<v>>\n");
    printf("    -n, --domain=STRING     simultaneously scan over one or two model params. STRING can be one of:\n");
    printf("                            1d: only one parameter\n");
    printf("    -o, --domainx=CHAR      sets the first domain of the moments. CHAR can be one of:\n");
    printf("                            a: fa; b: fb; m: mua; n: mub\n");
    printf("    -p, --logx=INT          choose between linear and logarithmic scale of the domainx\n");
    printf("                            0: linear; 1: logarithmic\n");
    printf("    -q, --points=INT        set the number of samples to generate between begin and end\n");
    printf("    -r, --beginx=FLOAT      set the starting value of the domainx to FLOAT\n");
    printf("    -s, --endx=FLOAT        set the end value of the domainx to FLOAT\n");
    printf("\n");
}

void parse_cla(int argc, char **argv)
{
    int c, itmp;

    while( (c = getopt_long(argc, argv, "a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s", options, NULL)) != EOF) {
        switch (c) {
            case 'a':
                h_fa = atof(optarg);
                cudaMemcpyToSymbol(d_fa, &h_fa, sizeof(float));
                break;
            case 'b':
                h_fb = atof(optarg);
                cudaMemcpyToSymbol(d_fb, &h_fb, sizeof(float));
                break;
            case 'c':
                h_mua = atof(optarg);
                cudaMemcpyToSymbol(d_mua, &h_mua, sizeof(float));
                break;
            case 'd':
                h_mub = atof(optarg);
                cudaMemcpyToSymbol(d_mub, &h_mub, sizeof(float));
                break;
            case 'e':
                h_comp = atoi(optarg);
                cudaMemcpyToSymbol(d_comp, &h_comp, sizeof(int));
                break;
            case 'f':
                itmp = atoi(optarg);
                cudaSetDevice(itmp);
                break;
            case 'g':
                h_block = atoi(optarg);
                break;
            case 'h':
                h_paths = atol(optarg);
                cudaMemcpyToSymbol(d_paths, &h_paths, sizeof(long));
                break;
            case 'i':
                h_periods = atol(optarg);
                break;
            case 'j':
                h_trans = atof(optarg);
                break;
            case 'k':
                h_spp = atoi(optarg);
                cudaMemcpyToSymbol(d_spp, &h_spp, sizeof(int));
                break;
            case 'l':
                h_samples = atoi(optarg);
                cudaMemcpyToSymbol(d_samples, &h_samples, sizeof(int));
                break;
            case 'm':
                if ( !strcmp(optarg, "moments") ) {
                    h_moments = 1;
                }
                break;
            case 'n':
                h_domain = optarg;
                break;
            case 'o':
                h_domainx = optarg[0]; 
                cudaMemcpyToSymbol(d_domainx, &h_domainx, sizeof(char));
                break;
            case 'p':
                h_logx = atoi(optarg);
                break;
            case 'q':
                h_points = atoi(optarg);
                cudaMemcpyToSymbol(d_points, &h_points, sizeof(int));
                break;
            case 'r':
                h_beginx = atof(optarg);
                break;
            case 's':
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

__global__ void init_dich(float *d_dx, int *d_dcd, int *d_dst, curandState *d_states)
//init dichotomous noise
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    float l_dx;
    curandState l_state;

    //cache model parameters in local variables
    l_state = d_states[idx];
    
    float l_fa, l_fb, l_mua, l_mub;
    int l_comp;
    
    l_fa = d_fa;
    l_fb = d_fb;
    l_mua = d_mua;
    l_mub = d_mub;
    l_comp = d_comp;

    long ridx = (idx/d_paths) % d_points;
    l_dx = d_dx[ridx];

    switch(d_domainx) {
        case 'a':
            l_fa = l_dx;
            if (l_comp == 1) {
                l_fb = -l_fa*l_mub/l_mua;
            }
            break;
        case 'b':
            l_fb = l_dx;
            if (l_comp == 1) {
                l_fa = -l_fb*l_mua/l_mub;
            } 
            break;
        case 'm':
            l_mua = l_dx;
            if (l_comp == 1) {
                l_mub = -l_fb*l_mua/l_fa;
            }
            break;
        case 'n':
            l_mub = l_dx;
            if (l_comp == 1) {
                l_mua = -l_fa*l_mub/l_fb;
            }
            break;
    }

    //step size
    float l_dt, taua, taub;
    int l_spp;

    l_spp = d_spp;

    taua = 1.0f/l_mua;
    taub = 1.0f/l_mub;

    if (taua < taub) {
        l_dt = taua/l_spp;
    } else {
        l_dt = taub/l_spp;
    }

    //jump countdown 
    int l_dcd, l_dst;

    float rn;
    rn = curand_uniform(&l_state);

    if (rn < 0.5f) {
        l_dst = 0;
        l_dcd = (int) floorf( -logf( curand_uniform(&l_state) )/l_mua/l_dt + 0.5f);
    } else {
        l_dst = 1;
        l_dcd = (int) floorf( -logf( curand_uniform(&l_state) )/l_mub/l_dt + 0.5f);
    }
   
    //write back noise state to the global memory
    d_dcd[idx] = l_dcd;
    d_dst[idx] = l_dst;
    d_states[idx] = l_state;
}

__device__ float adapted_jump_dich(int &ndcd, int dcd, int &ndst, int dst, float l_fa, float l_fb, float l_mua, float l_mub, float l_dt, curandState *l_state)
{
    if (dcd <= 0) {
        if (dst == 0) {
            ndst = 1; 
            ndcd = (int) floorf( -logf( curand_uniform(l_state) )/l_mub/l_dt + 0.5f );
            return l_fb*l_dt;
        } else {
            ndst = 0;
            ndcd = (int) floorf( -logf( curand_uniform(l_state) )/l_mua/l_dt + 0.5f );
            return l_fa*l_dt;
        }
    } else {
        ndcd = dcd - 1;
        if (dst == 0) {
            return l_fa*l_dt;
        } else {
            return l_fb*l_dt;
        }
    }
}

__device__ void predcorr(float &corrl_x, float l_x, curandState *l_state, \
                         int &ndcd, int dcd, int &ndst, int dst, float l_fa, float l_fb, float l_mua, float l_mub, float l_dt)
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

    corrl_x = l_x + 0.5f*(l_xt + l_xtt)*l_dt + adapted_jump_dich(ndcd, dcd, ndst, dst, l_fa, l_fb, l_mua, l_mub, l_dt, l_state); 
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

__global__ void run_moments(float *d_x, float *d_dx, int *d_dcd, int *d_dst, curandState *d_states)
//actual moments kernel
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    float l_x, l_dx; 
    int l_dcd, l_dst;
    curandState l_state;

    //cache path and model parameters in local variables
    l_x = d_x[idx];
    l_dcd = d_dcd[idx];
    l_dst = d_dst[idx];
    l_state = d_states[idx];
    
    float l_fa, l_fb, l_mua, l_mub;
    int l_comp;

    l_fa = d_fa;
    l_fb = d_fb;
    l_mua = d_mua;
    l_mub = d_mub;
    l_comp = d_comp;

    //run simulation for multiple values of the system parameters
    long ridx = (idx/d_paths) % d_points;
    l_dx = d_dx[ridx];

    switch(d_domainx) {
        case 'a':
            l_fa = l_dx;
            if (l_comp == 1) {
                l_fb = -l_fa*l_mub/l_mua;
            }
            break;
        case 'b':
            l_fb = l_dx;
            if (l_comp == 1) {
                l_fa = -l_fb*l_mua/l_mub;
            } 
            break;
        case 'm':
            l_mua = l_dx;
            if (l_comp == 1) {
                l_mub = -l_fb*l_mua/l_fa;
            }
            break;
        case 'n':
            l_mub = l_dx;
            if (l_comp == 1) {
                l_mua = -l_fa*l_mub/l_fb;
            }
            break;
    }

    //step size & number of steps
    float l_dt, taua, taub;
    int i, l_spp, l_samples;

    l_spp = d_spp;

    taua = 1.0f/l_mua;
    taub = 1.0f/l_mub;

    if (taua < taub) {
        l_dt = taua/l_spp;
    } else {
        l_dt = taub/l_spp;
    }

    l_samples = d_samples;

    for (i = 0; i < l_samples; i++) {
        predcorr(l_x, l_x, &l_state, l_dcd, l_dcd, l_dst, l_dst, l_fa, l_fb, l_mua, l_mub, l_dt);
    }

    //write back path parameters to the global memory
    d_x[idx] = l_x;
    d_dcd[idx] = l_dcd;
    d_dst[idx] = l_dst;
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
    cudaMalloc((void**)&d_dcd, size_i);
    cudaMalloc((void**)&d_dst, size_i);
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
    float sx, sxb, taua, taub, dt, tmp;
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

        //Dichotomous
        if (h_domainx == 'm') {
            taua = 1.0f/h_dx[j];
            taub = 1.0f/h_mub;

            if (h_comp) {
                tmp = 1.0f/(-h_fb*h_dx[j]/h_fa);
            } else {
                tmp = taub;
            }

            if (taua <= tmp) {
                dt = taua/h_spp;
            } else {
                dt = tmp/h_spp;
            }
        } else if (h_domainx == 'n') {
            taua = 1.0f/h_mua;
            taub = 1.0f/h_dx[j];

            if (h_comp) {
                tmp = 1.0f/(-h_fa*h_dx[j]/h_fb);
            } else {
                tmp = taua;
            }

            if (taub <= tmp) {
                dt = taub/h_spp;
            } else {
                dt = tmp/h_spp;
            }
        } else if (h_mua != 0.0f || h_mub != 0.0f) {
            taua = 1.0f/h_mua;
            taub = 1.0f/h_mub;

            if (taua < taub) {
                dt = taua/h_spp;
            } else {
                dt = taub/h_spp;
            }
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
    cudaFree(d_dcd);
    cudaFree(d_dst);
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

            init_dich<<<h_grid, h_block>>>(d_dx, d_dcd, d_dst, d_states);

            for (i = 0; i < h_steps; i += h_samples) {
                run_moments<<<h_grid, h_block>>>(d_x, d_dx, d_dcd, d_dst, d_states);
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
