// Check that __kitomp_reduce_num_partials works as expected. This is expected
// to return the number of available threads.
//
// RUN: KIT_NUM_THREADS=3 %exe | FileCheck %s
//
// CHECK: Partial reductions = 3

#include "openmp/kitomp.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitomp_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitomp_finalize(); }

int main(int argc, char* argv[]) {
  printf("Partial reductions = %ld\n", __kitomp_reduce_num_partials(1024));
  return 0;
}
