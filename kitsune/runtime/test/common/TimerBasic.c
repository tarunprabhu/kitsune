// The most basic timer test. This sleeps for 1ms and performs a sanity check
// on the recorded time span.
//
// RUN: %exe

#include "common/timer.h"
#include "kitrt.h"

#include <stdio.h>
#include <time.h>

__attribute__((constructor)) static void ctor(void) { __kitrt_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitrt_finalize(); }

int main(int argc, char *argv[]) {
  // This is 1ms.
  struct timespec duration;
  duration.tv_sec = 0;
  duration.tv_nsec = 1000000;

  TimePoint start = __kittimer_start();
  nanosleep(&duration, NULL);
  TimeSpan span =
      __kittimer_stop(start, /*timerID=*/42, /*threadID=*/0, "basic");

  // The span is not guaranteed to be exactly tv_nsec. But it should not be
  // less than the requested sleep duration.
  if (span >= (TimeSpan)duration.tv_nsec)
    return 0;
  return 1;
}
