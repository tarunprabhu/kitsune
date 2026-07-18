// REQUIRES: nvidia-gpu
//
// Check that __kitcuda_is_initialized works as expected.
//
// RUN: %exe | FileCheck %s
//
// CHECK: Before initialize: 0
// CHECK: After initialize: {{[1-9][0-9]*}}
// CHECK: Before finalize: {{[1-9][0-9]*}}
// CHECK: After finalize: 0

#include "cuda/kitcuda.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) {
  printf("Before initialize: %d\n", __kitcuda_is_initialized());
  __kitcuda_initialize();
  printf("After initialize: %d\n", __kitcuda_is_initialized());
}

__attribute__((destructor)) static void dtor(void) {
  printf("Before finalize: %d\n", __kitcuda_is_initialized());
  __kitcuda_finalize();
  printf("After finalize: %d\n", __kitcuda_is_initialized());
}

int main(int argc, char *argv[]) { return 0; }
