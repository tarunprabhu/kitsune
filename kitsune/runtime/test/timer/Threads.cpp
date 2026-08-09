// Check that the output of per-thread timers on a multi-threaded application is
// as expected. We use Kitsune's OpenMP runtime because it is guaranteed to
// be built. Also, omp_get_thread_num() returns an integer in
// [0, KIT_NUM_THREADS), so we can match against thread ids.
//
// RUN: env KIT_NUM_THREADS=3 %exe 2>&1 | FileCheck %s
//
// CHECK:      {
// CHECK-NEXT:   "main": {
// CHECK-NEXT:     "0": [
// CHECK-NEXT:       {{[0-9]+}}
// CHECK-NEXT:     ]
// CHECK-NEXT:   },
// CHECK-NEXT:   "thrd": {
// CHECK-NEXT:     "0": [
// CHECK-NEXT:       {{[0-9]+}}
// CHECK-NEXT:     ],
// CHECK-NEXT:     "1": [
// CHECK-NEXT:       {{[0-9]+}}
// CHECK-NEXT:     ],
// CHECK-NEXT:     "2": [
// CHECK-NEXT:       {{[0-9]+}}
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: }

#include "TestHelpers.h"
#include "openmp/kitomp.h"
#include "timer/timer.h"

#include <stdlib.h>

CTOR(RT_TIMER | RT_OPENMP)

static void thrdFn(uint64_t start, uint64_t end, void *args) {
  kitrt::KitTimerEpoch *e = __kittimer_start("thrd", __kitomp_thread_id());
  __kittimer_stop(e);
}

int main(int argc, char *argv[]) {
  kitrt::KitTimerEpoch *e = __kittimer_start("main", /*thread=*/0);
  __kitomp_launch(thrdFn, /*beg=*/0, /*end=*/3, /*args=*/NULL, /*argSize=*/0);
  __kittimer_stop(e);

  return 0;
}
