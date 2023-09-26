#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>

#include <kitsune.h>
#include "kitsune/llvm-gpu-abi/llvm-gpu.h"
#include "kitsune/llvm-gpu-abi/kitrt-cuda.h"

void random_matrix(float *I, int rows, int cols) {
  srand(7);
  for(int i = 0 ; i < rows ; i++) {
    for ( int j = 0 ; j < cols ; j++){
      I[i * cols + j] = rand()/(float)RAND_MAX ;
    }
  }
}

void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lamda> <no. of iter>\n", argv[0]);
  fprintf(stderr, "\t<rows>   - number of rows\n");
  fprintf(stderr, "\t<cols>    - number of cols\n");
  fprintf(stderr, "\t<y1> 	 - y1 value of the speckle\n");
  fprintf(stderr, "\t<y2>      - y2 value of the speckle\n");
  fprintf(stderr, "\t<x1>       - x1 value of the speckle\n");
  fprintf(stderr, "\t<x2>       - x2 value of the speckle\n");
  fprintf(stderr, "\t<lamda>   - lambda (0,1)\n");
  fprintf(stderr, "\t<no. of iter>   - number of iterations\n");
  exit(1);
}

int main(int argc, char* argv[])
{
  int rows, cols, size_I, size_R, niter = 10, iter, k;
  float *I, *J, q0sqr, tmp, meanROI,varROI ;
  float Jc, G2, L, num, den, qsqr;
  int *iN,*iS,*jE,*jW;
  float *dN,*dS,*dW,*dE;
  int r1, r2, c1, c2;
  float cN,cS,cW,cE;
  float *c, D;
  float lambda;
  int i, j;

  if (argc == 9) {
    rows = atoi(argv[1]); //number of rows in the domain
    cols = atoi(argv[2]); //number of cols in the domain

    if ((rows%16!=0) || (cols%16!=0)){
      fprintf(stderr, "rows and cols must be multiples of 16\n");
      exit(1);
    }
    r1   = atoi(argv[3]); //y1 position of the speckle
    r2   = atoi(argv[4]); //y2 position of the speckle
    c1   = atoi(argv[5]); //x1 position of the speckle
    c2   = atoi(argv[6]); //x2 position of the speckle
    lambda = atof(argv[7]); //Lambda value
    niter = atoi(argv[8]); //number of iterations
  } else if (argc == 1) {
    // run with default configuration...
    rows = 6400;
    cols = 6400;
    r1 = 0;
    r2 = 127;
    c1 = 0;
    c2 = 127;
    lambda = 0.5;
    niter = 10;
  } else {
    usage(argc, argv);
  }

  size_I = cols * rows;
  size_R = (r2-r1+1)*(c2-c1+1);

  I = (float *)malloc(sizeof(float) * size_I);
  J = (float *)malloc(sizeof(float) * size_I);
  c = (float *)malloc(sizeof(float) * size_I);

  iN = (int *)malloc(sizeof(unsigned int*) * rows);
  iS = (int *)malloc(sizeof(unsigned int*) * rows);
  jW = (int *)malloc(sizeof(unsigned int*) * cols);
  jE = (int *)malloc(sizeof(unsigned int*) * cols);

  dN = (float *)malloc(sizeof(float)* size_I) ;
  dS = (float *)malloc(sizeof(float)* size_I) ;
  dW = (float *)malloc(sizeof(float)* size_I) ;
  dE = (float *)malloc(sizeof(float)* size_I) ;

  for(int i=0; i < rows; i++) {
    iN[i] = i-1;
    iS[i] = i+1;
  }

  for(int j=0; j < cols; j++) {
    jW[j] = j-1;
    jE[j] = j+1;
  }

  iN[0] = 0;
  iS[rows-1] = rows-1;
  jW[0] = 0;
  jE[cols-1] = cols-1;

  printf("Randomizing the input matrix\n");
  random_matrix(I, rows, cols);

  for(int k = 0;  k < size_I; k++ ) {
    J[k] = (float)exp(I[k]) ;
  }

  printf("Start the SRAD main loop\n");
  auto start = std::chrono::steady_clock::now();

  for (iter=0; iter< niter; iter++){
    float sum, sum2;
    sum  = 0.0;
    sum2 =0.0;
    for (int i=r1; i<=r2; i++) {
      for (int j=c1; j<=c2; j++) {
	tmp   = J[i * cols + j];
	sum  += tmp ;
	sum2 += tmp*tmp;
      }
    }
    meanROI = sum / size_R;
    varROI  = (sum2 / size_R) - meanROI*meanROI;
    q0sqr   = varROI / (meanROI*meanROI);

    for(int i = 0 ; i < rows ; i++) {
      for (int j = 0; j < cols; j++) {
	int k = i * cols + j;
	float Jc = J[k];
	// directional derivates
	dN[k] = J[iN[i] * cols + j] - Jc;
	dS[k] = J[iS[i] * cols + j] - Jc;
	dW[k] = J[i * cols + jW[j]] - Jc;
	dE[k] = J[i * cols + jE[j]] - Jc;

	float G2 = (dN[k]*dN[k] + dS[k]*dS[k]
                    + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

	float L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

	float num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;
	float den  = 1 + (.25*L);
	float qsqr = num/(den*den);

	// diffusion coefficent (equ 33)
	den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
	c[k] = 1.0 / (1.0+den) ;

	// saturate diffusion coefficent
	if (c[k] < 0) {c[k] = 0;}
	else if (c[k] > 1) {c[k] = 1;}
      }
    }

    for(int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
	// current index
	int k = i * cols + j;

	// diffusion coefficent
	float cN = c[k];
	float cS = c[iS[i] * cols + j];
	float cW = c[k];
	float cE = c[i * cols + jE[j]];

	// divergence (equ 58)
	float D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
	// image update (equ 61)
	J[k] = J[k] + 0.25*lambda*D;
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  printf("Computation Done: %.12f s\n",
	 std::chrono::duration<double>(end-start).count());

  FILE *fp = fopen("srad-serial.dat", "wb");
  if (fp != NULL) {
    fwrite((void*)J, sizeof(float), size_I, fp);
    fclose(fp);
  }

  return 0;
}



