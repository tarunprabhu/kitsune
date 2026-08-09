// REQUIRES: kitsune-papi
//
// Check that unknown PAPI event names are handled as expected.
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Unknown event name 'PAPI_TOT_INS'
// CHECK: Unknown event name 'TOT_INS'
// CHECK: Event 'PAPI_TOT_INS' added to epoch 'nilakantha'
// CHECK: Unknown event name 'Inst'
// CHECK: Unknown event name 'insts'
//
// CHECK:      {
// CHECK-NEXT:   "nilakantha": {
// CHECK-NEXT:     "19": [
// CHECK-NEXT:       {"Instr completed": {{[0-9]+}}}
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: }

#include "TestHelpers.h"
#include "common/kitpapi.h"

CTOR(RT_PAPI | RT_SERIAL)

int main(int argc, char *argv[]) {
  (void)__kitpapi_start("nilakantha", /*thread=*/19, 5, "PAPI_TOT_INS",
                        "TOT_INS", "tot_ins", "Inst", "insts");
  return 0;
}
