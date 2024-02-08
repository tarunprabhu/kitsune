//
// Copyright(c) 2020 Triad National Security, LLC
// All rights reserved.
//
// This file is part of the kitsune / llvm project.  It is released under
// the LLVM license.
//
// Simple example of an element-wise vector sum.
// To enable kitsune+tapir compilation add the flags to a standard
// clang compilation:
//
//    * -ftapir=rt-target : the runtime ABI to target.
//
#include <cstdio>
#include <stdlib.h>
#include <kitsune.h>
#include "kitsune/timer.h"

using namespace std;
using namespace kitsune;

const size_t VEC_SIZE = 1024 * 1024 * 256;

void random_fill(float *data, size_t N) {
  for(size_t i = 0; i < N; ++i) 
    data[i] = rand() / (float)RAND_MAX;
}

int main (int argc, char* argv[]) {

  fprintf(stderr, "kitsune+tapir kokkos example: element-wise vector addition\n");
  
  float *A = new float[VEC_SIZE];
  float *B = new float[VEC_SIZE];
  float *C = new float[VEC_SIZE];

  random_fill(A, VEC_SIZE);
  random_fill(B, VEC_SIZE);
  
  timer t;  
  forall(size_t i = 0; i < VEC_SIZE; i++) 
    C[i] = A[i] + B[i];
  double loop_secs = t.seconds();

  fprintf(stderr, "(%s) %lf, %lf, %lf, %lf\n", 
          argv[0], C[0], C[VEC_SIZE/4], C[VEC_SIZE/2], C[VEC_SIZE-1]);   
  
  fprintf(stdout, "%lf\n", loop_secs);

  delete []A;
  delete []B;
  delete []C;

  return 0;
}
