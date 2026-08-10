// REQUIRES: kitsune-papi
//
// Check that the output of per-thread PAPI epochs on a multi-threaded
// application is as expected. We use Kitsune's openmp runtime because it is
// guaranteed to be built, and supports PAPI.
//
// RUN: env KIT_NUM_THREADS=3 %exe 2>&1 | FileCheck %s
//
// CHECK:      {
// CHECK-NEXT:   "morales": {
// CHECK-NEXT:     "0": [
// CHECK-NEXT:       {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}}
// CHECK-NEXT:     ],
// CHECK-NEXT:     "1": [
// CHECK-NEXT:       {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}}
// CHECK-NEXT:     ],
// CHECK-NEXT:     "2": [
// CHECK-NEXT:       {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}}
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: }

#include "TestHelpers.h"
#include "openmp/kitomp.h"
#include "papi/kitpapi.h"

#include <stddef.h>

CTOR(RT_PAPI | RT_OPENMP)

static void thrdFn(uint64_t start, uint64_t end, void *args) {
  KitPAPIEpoch *e =
      __kitpapi_start("morales", __kitomp_thread_id(), 2, "ins", "cyc");
  __kitpapi_stop(e);
}

int main(int argc, char *argv[]) {
  __kitomp_launch(thrdFn, /*beg=*/0, /*end=*/3, /*args=*/NULL, /*argSize=*/0);

  return 0;
}
