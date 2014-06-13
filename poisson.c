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
#include <sys/time.h>

#define PI 3.14159265358979f

//model
float d_Dp, d_lambda, d_mean; // poissonian noise

//simulation
float d_trans;
long d_paths, d_periods, d_steps;
int d_spp, d_samples;
float d_x, d_xb, d_dx, d_dt;

static struct option options[] = {
    {"Dp", required_argument, NULL, 'b'},
    {"lambda", required_argument, NULL, 'c'},
    {"mean", required_argument, NULL, 'i'},
    {"paths", required_argument, NULL, 'l'},
    {"periods", required_argument, NULL, 'm'},
    {"trans", required_argument, NULL, 'n'},
    {"spp", required_argument, NULL, 'o'},
    {"samples", required_argument, NULL, 'j'}
};

void usage(char **argv)
{
    printf("Usage: %s <params> \n\n", argv[0]);
    printf("Model params:\n");
    printf("    -b, --Dp=FLOAT          set the Poissonian noise intensity 'D_P' to FLOAT\n");
    printf("    -c, --lambda=FLOAT      set the Poissonian kicks frequency '\\lambda' to FLOAT\n");
    printf("    -i, --mean=FLOAT        if is nonzero, fix the mean value of Poissonian noise or dichotomous noise to FLOAT, matters only for domains p, l, a, b, m or n\n");
    printf("Simulation params:\n");
    printf("    -l, --paths=LONG        set the number of paths to LONG\n");
    printf("    -m, --periods=LONG      set the number of periods to LONG\n");
    printf("    -n, --trans=FLOAT       specify fraction FLOAT of periods which stands for transients\n");
    printf("    -o, --spp=INT           specify how many integration steps should be calculated for a single period of the driving force\n");
    printf("    -j, --samples=INT       specify how many integration steps should be calculated for a single kernel call\n");
    printf("\n");
}

void parse_cla(int argc, char **argv)
{
    float ftmp;
    int c, itmp;

    while( (c = getopt_long(argc, argv, "b:c:i:l:m:n:o:j", options, NULL)) != EOF) {
        switch (c) {
            case 'b':
		sscanf(optarg, "%f", &d_Dp);
                break;
            case 'c':
		sscanf(optarg, "%f", &d_lambda);
                break;
            case 'i':
		sscanf(optarg, "%f", &d_mean);
                break;
            case 'l':
		sscanf(optarg, "%ld", &d_paths);
                break;
            case 'm':
		sscanf(optarg, "%ld", &d_periods);
                break;
            case 'n':
		sscanf(optarg, "%f", &d_trans);
                break;
            case 'o':
		sscanf(optarg, "%d", &d_spp);
                break;
            case 'j':
		sscanf(optarg, "%d", &d_samples);
                break;
            }
    }
}

float drift(float l_x)
{
  if (-sinf(PI*l_x) < 0.0f)
    return -1.0f;
  else
    return 1.0f;
}

float u01()
//easy to extend for any library with better statistics/algorithms (e.g. GSL)
{
  return (float)rand()/RAND_MAX;
}

float adapted_jump_poisson(int *npcd, int pcd, float l_lambda, float l_Dp, float l_dt)
{
  float comp = sqrtf(l_Dp*l_lambda)*l_dt;
  
  if (pcd <= 0) {
    float ampmean = sqrtf(l_lambda/l_Dp);
    *npcd = (int) floorf( -logf( u01() )/l_lambda/l_dt + 0.5f );
    return -logf( u01() )/ampmean - comp;
  } 
  else {
    *npcd = pcd - 1;
    return -comp;
  }
}

void predcorr(float *corrl_x, float l_x, int *npcd, int pcd, float l_Dp, float l_lambda, float l_dt)
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

    *corrl_x = l_x + 0.5f*(l_xt + l_xtt)*l_dt + adapted_jump_poisson(npcd, pcd, l_lambda, l_Dp, l_dt);
}

void fold(float *nx, float x, float y, float *nfc, float fc)
//reduce periodic variable to the base domain
{
  *nx = x - floorf(x/y)*y;
  *nfc = fc + floorf(x/y)*y;
}

void run_moments()
//actual moments kernel
{
  long i;
  int sample;
  //cache path and model parameters in local variables
  //this is only to maintain original GPGPU code
  float l_x = d_x,
	l_xb;

  float l_Dp = d_Dp,
	l_lambda = d_lambda,
	l_mean = d_mean;

  //step size & number of steps
  float l_dt = 1.0f/l_lambda/d_spp;

  //store step size in global mem
  d_dt = l_dt;

  long steps_samples = d_steps/d_samples,
       sample_trigger = lrint(d_trans * steps_samples);

  //counters for folding
  float xfc = 0.0f;

  //jump countdowns
  int pcd = (int) floorf( -logf( u01() )/l_lambda/l_dt + 0.5f );
 
  for (i = 0; i < steps_samples; i++) {

    for (sample = 0; sample < d_samples; sample++) 
      predcorr(&l_x, l_x, &pcd, pcd, l_Dp, l_lambda, l_dt);

    //fold path parameters
    fold(&l_x, l_x, 2.0f, &xfc, xfc);

    if (i == sample_trigger) 
      l_xb = l_x + xfc;

  }

  //write back path parameters to the global memory
  d_x = l_x + xfc;
  d_xb = l_xb;
}

void prepare()
//prepare simulation
{
  //number of steps
  d_steps = d_periods*d_spp;

  //initialization of rng
  srand(time(NULL));
}

void initial_conditions()
//set initial conditions for path parameters
{
  d_x = 2.0f*u01() - 1.0f; //x in (-1,1]
}

float moments()
//calculate the first moment of v
{
  return (d_x - d_xb)/( (1.0f - d_trans)*d_steps*d_dt );
}

void print_params()
{
  printf("#Dp %e\n",d_Dp);
  printf("#lambda %e\n",d_lambda);
  printf("#mean %e\n",d_mean);
  printf("#paths %ld\n",d_paths);
  printf("#periods %ld\n",d_periods);
  printf("#trans %f\n",d_trans);
  printf("#spp %d\n",d_spp);
}

long long current_timestamp() {
  struct timeval te; 
  gettimeofday(&te, NULL); // get current time
  long long milliseconds = te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
  return milliseconds;
}

int main(int argc, char **argv)
{
  long long t0, te;
  double tsim;

  parse_cla(argc, argv);
  //print_params();

  if (0) usage(argv);
  prepare();
  
  //asymptotic long time average velocity <<v>>
  float av = 0.0f;
  int i;

  int dump_av = 2;
  printf("#[1]no_of_runs [2]cpu_time(milisec) [3]Gsteps/sec [4]<<v>>\n");
  t0 = current_timestamp();
  for (i = 0; i < d_paths; ++i){

    initial_conditions();
    run_moments();
    //av += moments();

    if (i == dump_av - 1){
      te = current_timestamp();
      tsim = te - t0;
      fprintf(stdout,"%d %lf %e %e\n", i+1, tsim, (i+1)*d_periods*d_spp*(1.0e-12)/tsim,av/(i+1));
      fflush(stdout);
      dump_av *= 2;
    }
  }

  return EXIT_SUCCESS;
}
