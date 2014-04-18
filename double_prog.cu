/*
 * Overdamped Brownian particle in symmetric piecewise linear potential
 *
 * \dot{x} = -V'(x) + Gaussian, Poissonian and dichotomous noise
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
__constant__ double d_Dg, d_Dp, d_lambda, d_mean, d_fa, d_fb, d_mua, d_mub;
__constant__ int d_comp;
double h_lambda, h_fa, h_fb, h_mua, h_mub, h_mean;
int h_comp;

//simulation
double h_trans;
int h_dev, h_block, h_grid, h_spp;
long h_paths, h_periods, h_threads, h_steps, h_trigger;
__constant__ int d_spp;
__constant__ long d_paths, d_steps, d_trigger;

//output
char *h_domain;
char h_domainx;
double h_beginx, h_endx;
int h_logx, h_points, h_moments;
__constant__ char d_domainx;
__constant__ int d_points;

//vector
double *h_x, *h_xb, *h_dx;
double *d_x, *d_xb, *d_dx;
unsigned int *h_seeds, *d_seeds;
curandState *d_states;

size_t size_f, size_ui, size_p;
curandGenerator_t gen;

static struct option options[] = {
    {"Dg", required_argument, NULL, 'a'},
    {"Dp", required_argument, NULL, 'b'},
    {"lambda", required_argument, NULL, 'c'},
    {"fa", required_argument, NULL, 'd'},
    {"fb", required_argument, NULL, 'e'},
    {"mua", required_argument, NULL, 'f'},
    {"mub", required_argument, NULL, 'g'},
    {"comp", required_argument, NULL, 'h'},
    {"mean", required_argument, NULL, 'i'},
    {"dev", required_argument, NULL, 'j'},
    {"block", required_argument, NULL, 'k'},
    {"paths", required_argument, NULL, 'l'},
    {"periods", required_argument, NULL, 'm'},
    {"trans", required_argument, NULL, 'n'},
    {"spp", required_argument, NULL, 'o'},
    {"mode", required_argument, NULL, 'p'},
    {"domain", required_argument, NULL, 'q'},
    {"domainx", required_argument, NULL, 'r'},
    {"logx", required_argument, NULL, 's'},
    {"points", required_argument, NULL, 't'},
    {"beginx", required_argument, NULL, 'u'},
    {"endx", required_argument, NULL, 'v'}
};

void usage(char **argv)
{
    printf("Usage: %s <params> \n\n", argv[0]);
    printf("Model params:\n");
    printf("    -a, --Dg=double          set the Gaussian noise intensity 'D_G' to double\n");
    printf("    -b, --Dp=double          set the Poissonian noise intensity 'D_P' to double\n");
    printf("    -c, --lambda=double      set the Poissonian kicks frequency '\\lambda' to double\n");
    printf("    -d, --fa=double          set the first state of the dichotomous noise 'F_a' to double\n");
    printf("    -e, --fb=double          set the second state of the dichotomous noise 'F_b' to double\n");
    printf("    -f, --mua=double         set the transition rate of the first state of dichotomous noise '\\mu_a' to double\n");
    printf("    -g, --mub=double         set the transition rate of the second state of dichotomous noise '\\mu_b' to double\n");
    printf("    -h, --comp=INT          choose between biased and unbiased Poissonian or dichotomous noise. INT can be one of:\n");
    printf("                            0: biased; 1: unbiased\n");
    printf("    -i, --mean=double        if is nonzero, fix the mean value of Poissonian noise or dichotomous noise to double, matters only for domains p, l, a, b, m or n\n");
    printf("Simulation params:\n");
    printf("    -j, --dev=INT           set the gpu device to INT\n");
    printf("    -k, --block=INT         set the gpu block size to INT\n");
    printf("    -l, --paths=LONG        set the number of paths to LONG\n");
    printf("    -m, --periods=LONG      set the number of periods to LONG\n");
    printf("    -n, --trans=double       specify fraction double of periods which stands for transients\n");
    printf("    -o, --spp=INT           specify how many integration steps should be calculated for a single period of the driving force\n");
    printf("Output params:\n");
    printf("    -p, --mode=STRING       sets the output mode. STRING can be one of:\n");
    printf("                            moments: the first moment <<v>>\n");
    printf("    -q, --domain=STRING     simultaneously scan over one or two model params. STRING can be one of:\n");
    printf("                            1d: only one parameter\n");
    printf("    -r, --domainx=CHAR      sets the first domain of the moments. CHAR can be one of:\n");
    printf("                            D: Dg; p: Dp; l: lambda; a: fa; b: fb; m: mua; n: mub\n");
    printf("    -s, --logx=INT          choose between linear and logarithmic scale of the domainx\n");
    printf("                            0: linear; 1: logarithmic\n");
    printf("    -t, --points=INT        set the number of samples to generate between begin and end\n");
    printf("    -u, --beginx=double      set the starting value of the domainx to double\n");
    printf("    -v, --endx=double        set the end value of the domainx to double\n");
    printf("\n");
}

void parse_cla(int argc, char **argv)
{
    double ftmp;
    int c, itmp;

    while( (c = getopt_long(argc, argv, "a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v", options, NULL)) != EOF) {
        switch (c) {
            case 'a':
                ftmp = atof(optarg);
                cudaMemcpyToSymbol(d_Dg, &ftmp, sizeof(double));
                break;
            case 'b':
                ftmp = atof(optarg);
                cudaMemcpyToSymbol(d_Dp, &ftmp, sizeof(double));
                break;
            case 'c':
                h_lambda = atof(optarg);
                cudaMemcpyToSymbol(d_lambda, &h_lambda, sizeof(double));
                break;
            case 'd':
                h_fa = atof(optarg);
                cudaMemcpyToSymbol(d_fa, &h_fa, sizeof(double));
                break;
            case 'e':
                h_fb = atof(optarg);
                cudaMemcpyToSymbol(d_fb, &h_fb, sizeof(double));
                break;
            case 'f':
                h_mua = atof(optarg);
                cudaMemcpyToSymbol(d_mua, &h_mua, sizeof(double));
                break;
            case 'g':
                h_mub = atof(optarg);
                cudaMemcpyToSymbol(d_mub, &h_mub, sizeof(double));
                break;
            case 'h':
                h_comp = atoi(optarg);
                cudaMemcpyToSymbol(d_comp, &h_comp, sizeof(int));
                break;
            case 'i':
                h_mean = atof(optarg);
                cudaMemcpyToSymbol(d_mean, &h_mean, sizeof(double));
                break;
            case 'j':
                itmp = atoi(optarg);
                cudaSetDevice(itmp);
                break;
            case 'k':
                h_block = atoi(optarg);
                break;
            case 'l':
                h_paths = atol(optarg);
                cudaMemcpyToSymbol(d_paths, &h_paths, sizeof(long));
                break;
            case 'm':
                h_periods = atol(optarg);
                break;
            case 'n':
                h_trans = atof(optarg);
                break;
            case 'o':
                h_spp = atoi(optarg);
                cudaMemcpyToSymbol(d_spp, &h_spp, sizeof(int));
                break;
            case 'p':
                if ( !strcmp(optarg, "moments") ) {
                    h_moments = 1;
                }
                break;
            case 'q':
                h_domain = optarg;
                break;
            case 'r':
                h_domainx = optarg[0]; 
                cudaMemcpyToSymbol(d_domainx, &h_domainx, sizeof(char));
                break;
            case 's':
                h_logx = atoi(optarg);
                break;
            case 't':
                h_points = atoi(optarg);
                cudaMemcpyToSymbol(d_points, &h_points, sizeof(int));
                break;
            case 'u':
                h_beginx = atof(optarg);
                break;
            case 'v':
                h_endx = atof(optarg);
                break;
            }
    }
}

__global__ void init_dev_rng(unsigned int *d_seeds, curandState *d_states)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(d_seeds[idx], idx, 0, &d_states[idx]);
}

__device__ double drift(double l_x)
{
    double l_y, l_f;

    l_y = fmod(l_x, 2.0);

    if (l_y < -1.0) {
        l_y += 2.0;
    } else if (l_y > 1.0) {
        l_y -= 2.0;
    }

    if (l_y >= -1.0 && l_y < 0.0) {
        l_f = 1.0;
    } else if (l_y >= 0.0 && l_y <= 1.0) {
        l_f = -1.0;
    }

    return l_f;
}

__device__ double diffusion(double l_Dg, double l_dt, curandState *l_state)
{
    if (l_Dg != 0.0) {
        double r = curand_uniform_double(l_state);
        if ( r <= 1.0/6 ) {
            return -sqrtf(6.0*l_Dg*l_dt);
        } else if ( r > 1.0/6 && r <= 2.0/6 ) {
            return sqrtf(6.0*l_Dg*l_dt);
        } else {
            return 0.0;
        }
    } else {
        return 0.0;
    }
}

__device__ double adapted_jump_poisson(int &npcd, int pcd, double l_lambda, double l_Dp, int l_comp, double l_dt, curandState *l_state)
{
    if (l_lambda != 0.0) {
        double comp = sqrtf(l_Dp*l_lambda)*l_dt;
        if (pcd <= 0) {
            double ampmean = sqrtf(l_lambda/l_Dp);
           
            npcd = (int) floor( -logf( curand_uniform_double(l_state) )/l_lambda/l_dt + 0.5 );

            if (l_comp) {
                return -logf( curand_uniform_double(l_state) )/ampmean - comp;
            } else {
                return -logf( curand_uniform_double(l_state) )/ampmean;
            }
        } else {
            npcd = pcd - 1;
            if (l_comp) {
                return -comp;
            } else {
                return 0.0;
            }
        }
    } else {
        return 0.0;
    }
}

__device__ double adapted_jump_dich(int &ndcd, int dcd, int &ndst, int dst, double l_fa, double l_fb, double l_mua, double l_mub, double l_dt, curandState *l_state)
{
    if (l_mua != 0.0 || l_mub != 0.0) {
        if (dcd <= 0) {
            if (dst == 0) {
                ndst = 1; 
                ndcd = (int) floor( -logf( curand_uniform_double(l_state) )/l_mub/l_dt + 0.5 );
                return l_fb;
            } else {
                ndst = 0;
                ndcd = (int) floor( -logf( curand_uniform_double(l_state) )/l_mua/l_dt + 0.5 );
                return l_fa;
            }
        } else {
            ndcd = dcd - 1;
            if (dst == 0) {
                return l_fa;
            } else {
                return l_fb;
            }
        }
    } else {
        return 0.0;
    }
}

__device__ void predcorr(double &corrl_x, double l_x, int &npcd, int pcd, curandState *l_state, \
                         double l_Dg, double l_Dp, double l_lambda, int l_comp, \
                         int &ndcd, int dcd, int &ndst, int dst, double l_fa, double l_fb, double l_mua, double l_mub, double l_dt)
/* simplified weak order 2.0 adapted predictor-corrector scheme
( see E. Platen, N. Bruti-Liberati; Numerical Solution of Stochastic Differential Equations with Jumps in Finance; Springer 2010; p. 503, p. 532 )
*/
{
    double l_xt, l_xtt, predl_x;

    l_xt = drift(l_x);

    predl_x = l_x + l_xt*l_dt + diffusion(l_Dg, l_dt, l_state);

    l_xtt = drift(predl_x);

    predl_x = l_x + 0.5*(l_xt + l_xtt)*l_dt + diffusion(l_Dg, l_dt, l_state);

    l_xtt = drift(predl_x);

    corrl_x = l_x + 0.5*(l_xt + l_xtt)*l_dt + adapted_jump_dich(ndcd, dcd, ndst, dst, l_fa, l_fb, l_mua, l_mub, l_dt, l_state)*l_dt + diffusion(l_Dg, l_dt, l_state) + adapted_jump_poisson(npcd, pcd, l_lambda, l_Dp, l_comp, l_dt, l_state);
}

__device__ void fold(double &nx, double x, double y, double &nfc, double fc)
//reduce periodic variable to the base domain
{
    nx = x - floor(x/y)*y;
    nfc = fc + floor(x/y)*y;
}

__global__ void run_moments(double *d_x, double *d_xb, double *d_dx, curandState *d_states)
//actual moments kernel
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    double l_x, l_xb, l_dx; 
    curandState l_state;

    //cache path and model parameters in local variables
    l_x = d_x[idx];
    l_xb = d_xb[idx];
    l_state = d_states[idx];

    double l_Dg, l_Dp, l_lambda, l_mean, l_fa, l_fb, l_mua, l_mub;
    int l_comp;

    l_Dg = d_Dg;
    l_Dp = d_Dp;
    l_lambda = d_lambda;
    l_comp = d_comp;
    l_mean = d_mean;
    l_fa = d_fa;
    l_fb = d_fb;
    l_mua = d_mua;
    l_mub = d_mub;

    //run simulation for multiple values of the system parameters
    long ridx = (idx/d_paths) % d_points;
    l_dx = d_dx[ridx];

    switch(d_domainx) {
        case 'D':
            l_Dg = l_dx;
            break;
        case 'p':
            l_Dp = l_dx;
            if (l_mean != 0.0) l_lambda = (l_mean*l_mean)/l_Dp;
            break;
        case 'l':
            l_lambda = l_dx;
            if (l_mean != 0.0) l_Dp = (l_mean*l_mean)/l_lambda;
            break;
        case 'a':
            l_fa = l_dx;
            if (l_comp == 1) {
                l_fb = -l_fa*l_mub/l_mua;
            } else if (l_mean != 0.0) {
                l_fb = (l_mean*(l_mua + l_mub) - l_fa*l_mub)/l_mua;
            }
            break;
        case 'b':
            l_fb = l_dx;
            if (l_comp == 1) {
                l_fa = -l_fb*l_mua/l_mub;
            } else if (l_mean != 0.0) {
                l_fa = (l_mean*(l_mua + l_mub) - l_fb*l_mua)/l_mub;
            }
            break;
        case 'm':
            l_mua = l_dx;
            if (l_comp == 1) {
                l_mub = -l_fb*l_mua/l_fa;
            } else if (l_mean != 0.0) {
                l_mub = (l_fb - l_mean)*l_mua/(l_mean - l_fa);
            }
            break;
        case 'n':
            l_mub = l_dx;
            if (l_comp == 1) {
                l_mua = -l_fa*l_mub/l_fb;
            } else if (l_mean != 0.0) {
                l_mua = (l_fa - l_mean)*l_mub/(l_mean - l_fb);
            }
            break;
    }

    //step size & number of steps
    double l_dt;
    long l_steps, l_trigger, i;

    if (l_lambda != 0.0) {
        l_dt = 1.0/l_lambda/d_spp;
    }

    if (l_mua != 0.0 || l_mub != 0.0) {
        double taua, taub;

        taua = 1.0/l_mua;
        taub = 1.0/l_mub;

        if (taua < taub) {
            l_dt = taua/d_spp;
        } else {
            l_dt = taub/d_spp;
        }
    }

    l_steps = d_steps;
    l_trigger = d_trigger;

    //counters for folding
    double xfc;
    
    xfc = 0.0;

    int pcd, dcd, dst;

    //jump countdowns
    if (l_lambda != 0.0) pcd = (int) floor( -logf( curand_uniform_double(&l_state) )/l_lambda/l_dt + 0.5 );

    if (l_mua != 0.0 || l_mub != 0.0) {
        double rn;
        rn = curand_uniform_double(&l_state);

        if (rn < 0.5) {
            dst = 0;
            dcd = (int) floor( -logf( curand_uniform_double(&l_state) )/l_mua/l_dt + 0.5);
        } else {
            dst = 1;
            dcd = (int) floor( -logf( curand_uniform_double(&l_state) )/l_mub/l_dt + 0.5);
        }
    }
    
    for (i = 0; i < l_steps; i++) {

        predcorr(l_x, l_x, pcd, pcd, &l_state, l_Dg, l_Dp, l_lambda, l_comp, \
                 dcd, dcd, dst, dst, l_fa, l_fb, l_mua, l_mub, l_dt);
        
        //fold path parameters
        if ( fabs(l_x) > 2.0 ) {
            fold(l_x, l_x, 2.0, xfc, xfc);
        }

        if (i == l_trigger) {
            l_xb = l_x + xfc;
        }

    }

    //write back path parameters to the global memory
    d_x[idx] = l_x + xfc;
    d_xb[idx] = l_xb;
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
    cudaMemcpyToSymbol(d_steps, &h_steps, sizeof(long));
     
    //host memory allocation
    size_f = h_threads*sizeof(double);
    size_ui = h_threads*sizeof(unsigned int);
    size_p = h_points*sizeof(double);

    h_x = (double*)malloc(size_f);
    h_seeds = (unsigned int*)malloc(size_ui);

    //create & initialize host rng
    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

    curandGenerate(gen, h_seeds, h_threads);
 
    //device memory allocation
    cudaMalloc((void**)&d_x, size_f);
    cudaMalloc((void**)&d_seeds, size_ui);
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
        cudaMemcpyToSymbol(d_trigger, &h_trigger, sizeof(long));

        h_xb = (double*)malloc(size_f);
        h_dx = (double*)malloc(size_p);

        double dxtmp = h_beginx;
        double dxstep = (h_endx - h_beginx)/h_points;

        long i;
        
        //set domainx
        for (i = 0; i < h_points; i++) {
            if (h_logx) {
                h_dx[i] = pow(10.0, dxtmp);
            } else {
                h_dx[i] = dxtmp;
            }
            dxtmp += dxstep;
        }
        
        cudaMalloc((void**)&d_xb, size_f);
        cudaMalloc((void**)&d_dx, size_p);
    
        cudaMemcpy(d_dx, h_dx, size_p, cudaMemcpyHostToDevice);
    }
}

void copy_to_dev()
{
    cudaMemcpy(d_x, h_x, size_f, cudaMemcpyHostToDevice);
    if (h_moments) {
        cudaMemcpy(d_xb, h_xb, size_f, cudaMemcpyHostToDevice);
    }
}

void copy_from_dev()
{
    cudaMemcpy(h_x, d_x, size_f, cudaMemcpyDeviceToHost);
    if (h_moments) {
        cudaMemcpy(h_xb, d_xb, size_f, cudaMemcpyDeviceToHost);
    }
}

void initial_conditions()
//set initial conditions for path parameters
{
    int i;

    curandGenerateUniformDouble(gen, h_x, h_threads); //x in (0,1]

    for (i = 0; i < h_threads; i++) {
        h_x[i] = 2.0*h_x[i] - 1.0; //x in (-1,1]
    }

    if (h_moments) {
        memset(h_xb, 0, size_f);
    }
    
    copy_to_dev();
}

void moments(double *av)
//calculate the first moment of v
{
    double sx, sxb, tmp, taua, taub, dt;
    int i, j;

    cudaMemcpy(h_x, d_x, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_xb, d_xb, size_f, cudaMemcpyDeviceToHost);

    for (j = 0; j < h_points; j++) {
        sx = 0.0;
        sxb = 0.0;

        for (i = 0; i < h_paths; i++) {
            sx += h_x[j*h_paths + i];
            sxb += h_xb[j*h_paths + i];
        }

        //Poissonian
        if (h_domainx == 'l') {
            dt = 1.0/h_dx[j]/h_spp;
        } else if (h_domainx == 'p' && h_mean != 0.0) {
            dt = 1.0/(h_mean*h_mean/h_dx[j])/h_spp;
        } else if (h_lambda != 0.0) {
            dt = 1.0/h_lambda/h_spp;
        }

        //Dichotomous
        if (h_domainx == 'm') {
            taua = 1.0/h_dx[j];
            taub = 1.0/h_mub;

            if (h_comp) {
                tmp = 1.0/(-h_fb*h_dx[j]/h_fa);
            } else if (h_mean != 0.0) {
                tmp = (h_fb - h_mean)*h_dx[j]/(h_mean - h_fa);
            } else {
                tmp = taub;
            }

            if (taua <= tmp) {
                dt = taua/h_spp;
            } else {
                dt = tmp/h_spp;
            }
        } else if (h_domainx == 'n') {
            taua = 1.0/h_mua;
            taub = 1.0/h_dx[j];

            if (h_comp) {
                tmp = 1.0/(-h_fa*h_dx[j]/h_fb);
            } else if (h_mean != 0.0) {
                tmp = (h_fa - h_mean)*h_dx[j]/(h_mean - h_fb);
            } else {
                tmp = taua;
            }

            if (taub <= tmp) {
                dt = taub/h_spp;
            } else {
                dt = tmp/h_spp;
            }
        } else if (h_mua != 0.0 || h_mub != 0.0) {
            taua = 1.0/h_mua;
            taub = 1.0/h_mub;

            if (taua < taub) {
                dt = taua/h_spp;
            } else {
                dt = taub/h_spp;
            }
        }
            
        av[j] = (sx - sxb)/( (1.0 - h_trans)*h_steps*dt )/h_paths;
    }
}

void finish()
//free memory
{

    free(h_x);
    
    curandDestroyGenerator(gen);
    cudaFree(d_x);
    cudaFree(d_states);
    
    if (h_moments) {
        free(h_xb);
        free(h_dx);

        cudaFree(d_xb);
        cudaFree(d_dx);
    }
}

int main(int argc, char **argv)
{
    parse_cla(argc, argv);
    if (!h_moments) {
        usage(argv);
        return -1;
    }

    prepare();
    
    initial_conditions();
    
    //asymptotic long time average velocity <<v>>
    if (h_moments) {
        double *av;
        int i;

        av = (double*)malloc(size_p);

        if ( !strcmp(h_domain, "1d") ) {
            run_moments<<<h_grid, h_block>>>(d_x, d_xb, d_dx, d_states);
            moments(av);

            printf("#%c <<v>>\n", h_domainx);
            for (i = 0; i < h_points; i++) {
                printf("%e %e\n", h_dx[i], av[i]);
            }   
        }

        free(av);
    }

    finish();

    return 0;
}
