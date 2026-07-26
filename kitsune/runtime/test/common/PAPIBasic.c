// REQUIRES: kitsune-papi
//
// Check very basic functionality of Kitsune's PAPI support.
//
// RUN: %exe 2>&1 | FileCheck %s --match-full-lines
//
// CHECK:      {
// CHECK-NEXT:   "dancaire": {
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
  // The total number of instructions and the total number of cycles ought to be
  // available on all platforms - one would think.
  KitPAPIEpoch *e = __kitpapi_new("dancaire", PAPI_TOT_INS, PAPI_TOT_CYC, 0);
  __kitpapi_start(e);
  __kitpapi_stop(e);

  return 0;
}
