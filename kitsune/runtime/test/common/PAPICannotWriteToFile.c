// REQUIRES: kitsune-papi
//
// Do something sensible when a PAPI counters file is provided, but it cannot be
// written to.
//
// RUN: env KIT_PAPI_FILE= %exe 2>&1 | FileCheck %s
//
// CHECK: Could not open file for writing

#include "common/kitpapi.h"
#include "kitrt.h"

#include "papi.h"

#include <stddef.h>

__attribute__((constructor)) static void ctor(void) {
  __kitrt_initialize();
  __kitpapi_initialize(NULL);
}

__attribute__((destructor)) static void dtor(void) {
  __kitpapi_finalize();
  __kitrt_finalize();
}

int main(int argc, char *argv[]) {
  KitPAPIEpoch *e = __kitpapi_start("lakme", /*thread=*/0, 2, "inst", "cyc");
  __kitpapi_stop(e);

  return 0;
}
