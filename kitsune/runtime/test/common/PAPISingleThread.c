// REQUIRES: kitsune-papi
//
// Check that the output of multiple PAPI epochs in a single thread is as
// expected.
//
// RUN: %exe 2>&1 | FileCheck %s --match-full-lines
//
// CHECK:      {
// CHECK-NEXT:   "escamilo": {
// CHECK-NEXT:     "0": [
// CHECK-NEXT:       {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}}
// CHECK-NEXT:     ]
// CHECK-NEXT:   },
// CHECK-NEXT:   "frasquita": {
// CHECK-NEXT:     "0": [
// CHECK-NEXT:       {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}}
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: }

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
  KitPAPIEpoch *e1 =
      __kitpapi_new("frasquita", /*thread=*/0, PAPI_TOT_INS, PAPI_TOT_CYC, 0);
  __kitpapi_start(e1);
  __kitpapi_stop(e1);

  KitPAPIEpoch *e2 =
      __kitpapi_new("escamilo", /*thread=*/0, PAPI_TOT_INS, PAPI_TOT_CYC, 0);
  __kitpapi_start(e2);
  __kitpapi_stop(e2);

  return 0;
}
