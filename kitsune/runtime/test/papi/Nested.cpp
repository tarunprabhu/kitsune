// REQUIRES: kitsune-papi
//
// Nested calls to __kitpapi_start are not allowed. We don't fail if the user
// does this, but check that non-fatal errors are emitted. The output will be
// printed, The counters for the inner event should be 0.
//
// RUN: %exe 2>&1 | FileCheck %s
//
// CHECK: Could not start PAPI counters
// CHECK: Could not stop PAPI counters
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

#include "TestHelpers.h"
#include "papi/kitpapi.h"

CTOR(RT_PAPI)

int main(int argc, char *argv[]) {
  KitPAPIEpoch *eo = __kitpapi_start("pastia", /*thread=*/0, 1, "ins");
  KitPAPIEpoch *ei = __kitpapi_start("jose", /*thread=*/0, 1, "ins");
  __kitpapi_stop(ei);
  __kitpapi_stop(eo);

  return 0;
}
