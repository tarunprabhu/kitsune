// REQUIRES: kitsune-papi
//
// Check very basic functionality of Kitsune's PAPI support.
//
// RUN: %exe 2>&1 | FileCheck %s --match-full-lines
//
// CHECK:      {
// CHECK-NEXT:   "dancaire": {
// CHECK-NEXT:     "67": [
// CHECK-NEXT:       {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}}
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: }

#include "TestHelpers.h"
#include "papi/kitpapi.h"

CTOR(RT_PAPI)

int main(int argc, char *argv[]) {
  // The total number of instructions and the total number of cycles ought to be
  // available on all platforms - one would think.
  kitrt::KitPAPIEpoch *e =
      __kitpapi_start("dancaire", /*thread=*/67, 2, "ins", "cyc");
  __kitpapi_stop(e);

  return 0;
}
