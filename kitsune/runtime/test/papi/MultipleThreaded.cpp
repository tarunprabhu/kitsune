// REQUIRES: kitsune-papi
//
// If more than one threaded CPU runtime has been enabled, PAPI cannot be
// initialized.
//
// RUN: KIT_VERBOSE=1 not %exe 2>&1 | FileCheck %s
//
// CHECK: PAPI does not support multiple threaded CPU runtimes
// CHECK-NOT: Initializing Kitsune runtime (papi)
// CHECK-NOT: Initialized Kitsune runtime (papi)
// CHECK-NOT: {{^[{]}}

#include "TestHelpers.h"
#include "papi/kitpapi.h"

CTOR(RT_PAPI | RT_OPENMP | RT_PTHREADS)

int main(int argc, char *argv[]) {
  KitPAPIEpoch *e = __kitpapi_start("figaro", /*thread=*/0, 1, "ins");
  __kitpapi_stop(e);

  return 0;
}
