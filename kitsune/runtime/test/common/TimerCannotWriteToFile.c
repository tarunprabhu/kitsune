// Do something sensible when a timings file is provided, but it cannot be
// written to.
//
// RUN: env KIT_TIMING_FILE= %exe 2>&1 | FileCheck %s
//
// CHECK: Could not open file for writing

#include "kitrt.h"

__attribute__((constructor)) static void ctor(void) { __kitrt_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitrt_finalize(); }

int main(int argc, char *argv[]) {
  TimePoint tick = __kittimer_start();
  __kittimer_stop(tick, /*timer-id=*/11, /*thread-id=*/0, "no-file");

  return 0;
}
