// REQUIRES: kitsune-papi
//
// Nested calls to __kitpapi_start are not allowed. We don't fail if the user
// does this, but check that non-fatal errors are emitted. The output will be
// printed, The counters for the inner event should be 0.
//
// RUN: %exe 2>&1 | FileCheck %s
//
// CHECK: Could not start PAPI counters
// CHECK: Could not read final values of PAPI counters
// CHECK:      {
// CHECK-NEXT:   "jose": {
// CHECK-NEXT:     "0": [
// CHECK-NEXT:       {"Instr completed": 0}
// CHECK-NEXT:     ]
// CHECK-NEXT:   },
// CHECK-NEXT:   "pastia": {
// CHECK-NEXT:     "0": [
// CHECK-NEXT:       {"Instr completed": {{[0-9]+}}}
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
  KitPAPIEpoch *eo = __kitpapi_new("pastia", PAPI_TOT_INS, 0);
  __kitpapi_start(eo);

  KitPAPIEpoch *ei = __kitpapi_new("jose", PAPI_TOT_INS, 0);
  __kitpapi_start(ei);
  __kitpapi_stop(ei);

  __kitpapi_stop(eo);
  return 0;
}
