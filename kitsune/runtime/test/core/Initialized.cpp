// Check that __kitrt_initialized works as expected.
//
// RUN: %exe | FileCheck %s
//
// CHECK: Before initialize: false
// CHECK: After initialize: true
// CHECK: Before finalize: true
// CHECK: After finalize: false

#include "TestHelpers.h"
#include "kitrt.h"

#include <stdio.h>

const KitRTInitOptions initOpts{RT_COMMON};

__attribute__((constructor)) static void ctor(void) {
  printf("Before initialize: %s\n", BOOLSTR(__kitrt_initialized()));
  __kitrt_initialize(&initOpts);
  printf("After initialize: %s\n", BOOLSTR(__kitrt_initialized()));
}

__attribute__((destructor)) static void dtor(void) {
  printf("Before finalize: %s\n", BOOLSTR(__kitrt_initialized()));
  __kitrt_finalize(&initOpts);
  printf("After finalize: %s\n", BOOLSTR(__kitrt_initialized()));
}

MAIN
