// The most basic timer test. This sleeps for 1ms and performs a sanity check
// on the recorded time span.
//
// RUN: %exe 2>&1 | FileCheck %s
//
// CHECK:      {
// CHECK-NEXT:   "basic": {
// CHECK-NEXT:     "47": [
// CHECK-NEXT:       {{[0-9]+}}
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: }

#include "TestHelpers.h"
#include "timer/timer.h"

#include <stdio.h>
#include <time.h>

CTOR(RT_TIMER)

int main(int argc, char *argv[]) {
  // This is 1ms.
  struct timespec duration;
  duration.tv_sec = 0;
  duration.tv_nsec = 1000000;

  KitTimerEpoch *e = __kittimer_start("basic", /*threadID=*/47);
  nanosleep(&duration, NULL);
  KitTimeSpan span = __kittimer_stop(e);

  // The span is not guaranteed to be exactly tv_nsec. But it should not be
  // less than the requested sleep duration.
  if (span >= (KitTimeSpan)duration.tv_nsec)
    return 0;
  return 1;
}
