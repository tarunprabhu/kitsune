// Do something sensible when a timings file is provided, but it cannot be
// written to.
//
// RUN: env KIT_TIMING_FILE= %exe 2>&1 | FileCheck %s
//
// CHECK: Could not open file for writing

#include "TestHelpers.h"
#include "timer/timer.h"

CTOR(RT_TIMER)

int main(int argc, char *argv[]) {
  KitTimerEpoch *e = __kittimer_start("no-file", /*thread=*/0);
  __kittimer_stop(e);

  return 0;
}
