// REQUIRES: kitsune-papi
//
// Kitsune does not have a serial runtime, but if code compiled with the serial
// tapir target is run, it should support PAPI.
//
// RUN: %exe | FileCheck %s
//
// CHECK: Before initialize: false
// CHECK: After initialize: true

#include "TestHelpers.h"

#include "papi.h"

#include <stdio.h>

const KitRTInitOptions initOpts{RT_PAPI | RT_SERIAL};

__attribute__((constructor)) static void ctor(void) {
  printf("Before initialize: %s\n", BOOLSTR(PAPI_is_initialized()));
  __kitrt_initialize(&initOpts);
  printf("After initialize: %s\n", BOOLSTR(PAPI_is_initialized()));
}

__attribute__((destructor)) static void dtor(void) {
  __kitrt_finalize(&initOpts);
}

int main(int argc, char *argv[]) { return 0; }
