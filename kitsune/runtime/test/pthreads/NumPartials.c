// Check that __kitpthr_reduce_num_partials works as expected. This is expected
// to return the number of available threads.
//
// RUN: KIT_NUM_THREADS=3 %exe | FileCheck %s
//
// CHECK: Partial reductions = 3

#include "pthreads/kitpthr.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitpthr_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitpthr_finalize(); }

int main(int argc, char* argv[]) {
  printf("Partial reductions = %ld\n", __kitpthr_reduce_num_partials(1024));
  return 0;
}
