#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <complex.h>

/* Include NFFT3 library header. */
#include "nfft3.h"

#define PLANS_START 10 /* initial number of plans */
#define CMD_LEN_MAX 20 /* maximum length of command argument */
/* global flags */
#define NFSOFT_MEX_FIRST_CALL (1U << 0)
#define KTRIG(x) (x)
static unsigned short gflags = NFSOFT_MEX_FIRST_CALL;
static nfsoft_plan** plans = NULL; /* plans */
static unsigned int plans_num_allocated = 0;
static int n_max = -1; /* maximum degree precomputed */
static char cmd[CMD_LEN_MAX];


static inline int mkplan(){
  int i = 0;
  while (i < plans_num_allocated && plans[i] != 0) i++;
  if (i == plans_num_allocated)
  {
    int l;

    nfsoft_plan** plans_old = plans;
    plans = nfft_malloc((plans_num_allocated+PLANS_START)*sizeof(nfsoft_plan*));
    for (l = 0; l < plans_num_allocated; l++)
      plans[l] = plans_old[l];
    for (l = plans_num_allocated; l < plans_num_allocated+PLANS_START; l++)
      plans[l] = 0;
    if (plans_num_allocated > 0)
      nfft_free(plans_old);
    plans_num_allocated += PLANS_START;
  }
  plans[i] = nfft_malloc(sizeof(nfsoft_plan));
  return i;
}
// Inputs for ODF and their representations:
// euler_ang - euler angles
// or_weights - orientation weights
// kernel_coef - kernel coefficients
void ODF(complex *c_hat,int euler_ang_total_per_row,int euler_ang_cols,double euler_ang_temp [], double or_weights[] , double kernel_coef[], int kernels_rows){
    // reshape euler_ang array from 1d to 2d
    double euler_ang[3][3*euler_ang_total_per_row];
    for(int j =0; j < (3*euler_ang_total_per_row);j++ ){
        if ((j/euler_ang_total_per_row) < 1){
        euler_ang[0][j] = euler_ang_temp[j];
        }
        else
            if((j/(euler_ang_total_per_row*2) > 1) && (j/(euler_ang_total_per_row*2) < 2 )){
            euler_ang[1][j/euler_ang_total_per_row*2] = euler_ang_temp[j];
        }
        else
            if((j/(euler_ang_total_per_row*3) > 2) && (j/(euler_ang_total_per_row*3) < 3 )){
            euler_ang[2][j/euler_ang_total_per_row*3] = euler_ang_temp[j];
            }

    }



    int nfsoft_flags = 16; // Parameter taken from mtex
    //Calculation
    int N = kernels_rows -1;
    int M = euler_ang_total_per_row;
    int flags_int = nfsoft_flags;
    int nfft_flags_int = 0;
    int nfft_cutoff = 4;
    int fpt_kappa = 1000;
    int fftw_size = 2 * ceil(1.5*(N+2));
    unsigned flags = (unsigned) flags_int;
    unsigned nfft_flags = (unsigned) nfft_flags_int;
    static const double K2PI =6.2831853071795864769252867665590057683943388;
    int i = mkplan();
    nfsoft_init_guru_advanced(plans[i], N, M, flags | NFSOFT_MALLOC_X | NFSOFT_MALLOC_F | NFSOFT_MALLOC_F_HAT,
	nfft_flags | PRE_PHI_HUT | PRE_PSI | MALLOC_F_HAT | FFTW_INIT, nfft_cutoff, fpt_kappa, fftw_size);
    plans[i]->p_nfft.f = plans[i]->f;
    plans[i]->p_nfft.x = plans[i]->x;
    double *x = &euler_ang[0][0];
        for (int j = 0; j < plans[i]->M_total; j++)
    {
    plans[i]->p_nfft.x[3*j]   = x[3*j+2] / K2PI;  // gamma
    plans[i]->p_nfft.x[3*j+1] = x[3*j]   / K2PI;  // alpha
    plans[i]->p_nfft.x[3*j+2] = x[3*j+1] / K2PI;  // beta
    }
    nfsoft_precompute(plans[i]);

    // set_f

    double *f_real = &or_weights[0], *f_imag=0;

    for (int j = 0; j < plans[i]->M_total;j++)
		{
	        plans[i]->f[j] = f_real[j];
		}

    //adjoint
    nfsoft_adjoint(plans[i]);


    //get_f_hat
    int glo1 = 0;
    int fh_size = NFSOFT_F_HAT_SIZE(N);
//    complex c_hat [fh_size];
    for (int k = -N; k <= N; k++)
	{
	  for (int m = -N; m <= N; m++)
	  {
		int max = (abs(m) > abs(k) ? abs(m) : abs(k));
		for (int j = max; j <= N; j++)		// j = polynomial degree
		{
		int my_ind = NFSOFT_F_HAT_SIZE(j-1) + (m+j)*(2*j+1) + (k+j);
		c_hat[my_ind] = plans[i]->f_hat[glo1];
		glo1++;
		}
	  }
	};


	//finalize
	nfsoft_finalize(plans[i]);
	nfft_free(plans[i]);
	plans[i] = 0;
//  stores coefficents into into the c_hat pointer can be accessed in main;


}

// Not Completed yet, output is not corrected and
// is not formated for use as a python function yet.
void eval_coefficients(complex *c_hat){
    int L = 29;
    int length_g = 10;

    double phi2[10];
    double phi1[10];
    double PHI[10];

  for (int i = 0; i < 10; i++){
        phi1[i] = 0 * (2*3.14);
        phi2[i] = 0 * (2*3.14);
        PHI[i] = 0 * (3.14);
    }
    double euler_ang [3][length_g];
    double output [length_g];
    for(int j =0; j < 10; j++){

        euler_ang[0][j] = phi1[j];
        euler_ang[1][j] = PHI[j];
        euler_ang[2][j] = phi2[j];

    }


	int N = L;
	int M = length_g;
	int flags_int = 16;
	unsigned flags = (unsigned) flags_int;
	int nfft_flags_int = 0;
	unsigned nfft_flags = (unsigned) nfft_flags_int;
	int nfft_cutoff = 4;
	int fpt_kappa = 1000;
	int fftw_size = 2*ceil(1.5*N);
	static const double K2PI =6.2831853071795864769252867665590057683943388;
	int i = mkplan();
	nfsoft_init_guru_advanced(plans[i], N, M, flags | NFSOFT_MALLOC_X | NFSOFT_MALLOC_F | NFSOFT_MALLOC_F_HAT,
	nfft_flags | PRE_PHI_HUT | PRE_PSI | MALLOC_F_HAT | FFTW_INIT, nfft_cutoff, fpt_kappa, fftw_size);
    plans[i]->p_nfft.f = plans[i]->f;
    plans[i]->p_nfft.x = plans[i]->x;
	// set_x
	double *x =  &euler_ang[0][0];
	for (int j = 0; j < plans[i]->M_total; j++)
    {
    plans[i]->p_nfft.x[3*j]   = x[3*j+2] / K2PI;  // gamma
    plans[i]->p_nfft.x[3*j+1] = x[3*j]   / K2PI;  // alpha
    plans[i]->p_nfft.x[3*j+2] = x[3*j+1] / K2PI;  // beta
    }
    nfsoft_precompute(plans[i]);

    //set_f_hat
    int fh_size = NFSOFT_F_HAT_SIZE(N);
    int glo1 = 0;
	for (int k = -N; k <= N; k++)
	{
	for (int m = -N; m <= N; m++)
	{
		int max = (abs(m) > abs(k) ? abs(m) : abs(k));
		for (int j = 0; j <= N; j++)		// j = polynomial degree
		{
		int my_ind = NFSOFT_F_HAT_SIZE(j-1) + (m+j)*(2*j+1) + (k+j);
			plans[i]->f_hat[j] = c_hat[j] ;
		glo1++;
		}
	}
	}

	//trafo or transform
	nfsoft_trafo(plans[i]);

	//get_f
    for (int j = 0; j < plans[i]->M_total; j++)
        {
          output[j] = creal(plans[i]->f[j]);

        }

    //finalize
	nfsoft_finalize(plans[i]);
	nfft_free(plans[i]);
	plans[i] = 0;
//	return output;
      }



int main()
{
    double kernel_coef [29] = {1,
    2.935416583,
    4.683960583,
    6.142961241,
    7.239253631,
    7.934902041,
    8.2281961,
    8.150161,
    7.75751989,
    7.123506819,
    6.328087686,
    5.449011711,
    4.554755252,
    3.699938907,
    2.923309935,
    2.247978305,
    1.683332713,
    1.227958327,
    0.8729111345,
    0.6048326657,
    0.4085639071,
    0.269093971,
    0.1728259417,
    0.1082433705,
    0.06611410068,
    0.03938115246,
    0.02287571894,
    0.012957877,
    0.007157107717,
    };

    double phi2[100];
    double phi1[100];
    double PHI[100];


  for (int i = 0; i < 100; i++){
        phi1[i] = 0 * (2*3.14);
        phi2[i] = 0 * (2*3.14);
        PHI[i] = 0 * (3.14);
    }


    double euler_ang [300];
    for(int j =0; j < 100; j++){

        euler_ang[j] = phi1[j];

    }
    for(int j =0; j < 100; j++){


        euler_ang[j+100] = PHI[j];

    }
    for(int j =0; j < 100; j++){
        euler_ang[j+200] = phi2[j];
    }

    double or_weights [100];
    for(int i = 0; i < 100; i++){
        or_weights[i] = (1.0/48.0)/100.0;
        }

    int kernels_rows = sizeof(kernel_coef) / sizeof(kernel_coef[0]); // returns rows
    int euler_ang_cols = sizeof(euler_ang) / sizeof(euler_ang[0]); // returns rows
    int euler_ang_total_per_row = euler_ang_cols/3;  // returns number of columns per array


//    int L = kernels_rows + 1;
//    int fh_size = (L * ((2*L)-1)*((2*L)+1))/3;
    int fh_size = NFSOFT_F_HAT_SIZE(sizeof(kernel_coef) / sizeof(double) -1);
    complex c_hat [fh_size];
    ODF(c_hat,euler_ang_total_per_row,euler_ang_cols,euler_ang,or_weights,kernel_coef,kernels_rows);

    eval_coefficients(c_hat);

























// Original Testing Version of ODF before being converted into a function
// to be used in python
//
//
//    int nfsoft_flags = 16; // Parameter taken from mtex
//
//    //Calculation
//    int N = sizeof(kernel_coef) / sizeof(double) -1;
//    int M = sizeof(euler_ang[0]) / sizeof(double);
//    int flags_int = nfsoft_flags;
//    int nfft_flags_int = 0;
//    int nfft_cutoff = 4;
//    int fpt_kappa = 1000;
//    int fftw_size = 2 * ceil(1.5*(N+2));
//    unsigned flags = (unsigned) flags_int;
//    unsigned nfft_flags = (unsigned) nfft_flags_int;
//    static const double K2PI =6.2831853071795864769252867665590057683943388;
//    int i = mkplan();
//    nfsoft_init_guru_advanced(plans[i], N, M, flags | NFSOFT_MALLOC_X | NFSOFT_MALLOC_F | NFSOFT_MALLOC_F_HAT,
//	nfft_flags | PRE_PHI_HUT | PRE_PSI | MALLOC_F_HAT | FFTW_INIT, nfft_cutoff, fpt_kappa, fftw_size);
//    plans[i]->p_nfft.f = plans[i]->f;
//    plans[i]->p_nfft.x = plans[i]->x;
//    double *x = &euler_ang[0][0];
//        for (int j = 0; j < plans[i]->M_total; j++)
//    {
//    plans[i]->p_nfft.x[3*j]   = x[3*j+2] / K2PI;  // gamma
//    plans[i]->p_nfft.x[3*j+1] = x[3*j]   / K2PI;  // alpha
//    plans[i]->p_nfft.x[3*j+2] = x[3*j+1] / K2PI;  // beta
//    }
//    nfsoft_precompute(plans[i]);
//
//    // set_f
//
//    double *f_real = &or_weights[0], *f_imag=0;
//
//    for (int j = 0; j < plans[i]->M_total;j++)
//		{
//	        plans[i]->f[j] = f_real[j];
//		}
//
//    //adjoint
//    nfsoft_adjoint(plans[i]);
//
//
//    //get_f_hat
//    int glo1 = 0;
//    int fh_size = NFSOFT_F_HAT_SIZE(N);
//    complex c_hat [fh_size];
//    for (int k = -N; k <= N; k++)
//	{
//	  for (int m = -N; m <= N; m++)
//	  {
//		int max = (abs(m) > abs(k) ? abs(m) : abs(k));
//		for (int j = max; j <= N; j++)		// j = polynomial degree
//		{
//		int my_ind = NFSOFT_F_HAT_SIZE(j-1) + (m+j)*(2*j+1) + (k+j);
//		c_hat[my_ind] = plans[i]->f_hat[glo1];
//		glo1++;
//		}
//	  }
//	};
//
//
//	//finalize
//	nfsoft_finalize(plans[i]);
//	nfft_free(plans[i]);
//	plans[i] = 0;
//    return 0 ;

}


