// REQUIRES: kitsune-papi
//
// Do something sensible when a PAPI counters file is provided, but it cannot be
// written to.
//
// RUN: env KIT_PAPI_FILE= %exe 2>&1 | FileCheck %s
//
// CHECK: Could not open file for writing

#include "TestHelpers.h"
#include "papi/kitpapi.h"

CTOR(RT_PAPI)

int main(int argc, char *argv[]) {
  KitPAPIEpoch *e = __kitpapi_start("lakme", /*thread=*/0, 2, "inst", "cyc");
  __kitpapi_stop(e);

  return 0;
}
