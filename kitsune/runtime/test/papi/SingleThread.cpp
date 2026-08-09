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

#include "TestHelpers.h"
#include "papi/kitpapi.h"

CTOR(RT_PAPI | RT_SERIAL)

int main(int argc, char *argv[]) {
  kitrt::KitPAPIEpoch *e1 =
      __kitpapi_start("frasquita", /*thread=*/0, 2, "ins", "cyc");
  __kitpapi_stop(e1);

  kitrt::KitPAPIEpoch *e2 =
      __kitpapi_start("escamilo", /*thread=*/0, 2, "ins", "cyc");
  __kitpapi_stop(e2);

  return 0;
}
