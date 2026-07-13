// Check that __kitocilk_reduce_num_partials works as expected. This is expected
// to return the number of available workers.
//
// RUN: KIT_NUM_THREADS=3 %exe | FileCheck %s
//
// CHECK: Partial reductions = 3

#include "opencilk/kitocilk.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitocilk_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitocilk_finalize(); }

int main(int argc, char *argv[]) {
  printf("Partial reductions = %ld\n", __kitocilk_reduce_num_partials(1024));
  return 0;
}
