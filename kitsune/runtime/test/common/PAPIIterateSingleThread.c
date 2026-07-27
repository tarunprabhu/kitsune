// REQUIRES: kitsune-papi
//
// Check that the output of PAPI epochs started and stopped multiple times is as
// expected.
//
// RUN: %exe 2>&1 | FileCheck %s
//
// CHECK:     {
// CHECK-NEXT:  "carmen": {
// CHECK-NEXT:    "0": [
// CHECK-NEXT:      {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}},
// CHECK-NEXT:      {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}},
// CHECK-NEXT:      {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}}
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:}

#include "common/kitpapi.h"
#include "kitrt.h"

#include "papi.h"

__attribute__((constructor)) static void ctor(void) {
  __kitrt_initialize();
  __kitpapi_initialize(NULL);
}

__attribute__((destructor)) static void dtor(void) {
  __kitpapi_finalize();
  __kitrt_finalize();
}

int main(int argc, char *argv[]) {
  for (unsigned i = 0; i < 3; ++i) {
    KitPAPIEpoch *e =
        __kitpapi_new("carmen", /*thread=*/0, PAPI_TOT_INS, PAPI_TOT_CYC, 0);
    __kitpapi_start(e);
    __kitpapi_stop(e);
  }

  return 0;
}
