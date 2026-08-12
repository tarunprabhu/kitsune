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
// CHECK-NEXT:       {
// CHECK-SAME:         "total": [[M:[0-9]+]],
// CHECK-SAME:         "visits": 1,
// CHECK-SAME:         "min": [[M]],
// CHECK-SAME:         "mean": [[M]],
// CHECK-SAME:         "max": [[M]]
// CHECK-SAME:       }
// CHECK-NEXT:     ]
// CHECK-NEXT:   },
// CHECK-NEXT:   "thrd": {
// CHECK-NEXT:     "0": [
// CHECK-NEXT:       {
// CHECK-SAME:         "total": [[T0:[0-9]+]],
// CHECK-SAME:         "visits": 1,
// CHECK-SAME:         "min": [[T0]],
// CHECK-SAME:         "mean": [[T0]],
// CHECK-SAME:         "max": [[T0]]
// CHECK-SAME:       }
// CHECK-NEXT:     ],
// CHECK-NEXT:     "1": [
// CHECK-NEXT:       {
// CHECK-SAME:         "total": [[T1:[0-9]+]],
// CHECK-SAME:         "visits": 1,
// CHECK-SAME:         "min": [[T1]],
// CHECK-SAME:         "mean": [[T1]],
// CHECK-SAME:         "max": [[T1]]
// CHECK-SAME:       }
// CHECK-NEXT:     ],
// CHECK-NEXT:     "2": [
// CHECK-NEXT:       {
// CHECK-SAME:         "total": [[T2:[0-9]+]],
// CHECK-SAME:         "visits": 1,
// CHECK-SAME:         "min": [[T2]],
// CHECK-SAME:         "mean": [[T2]],
// CHECK-SAME:         "max": [[T2]]
// CHECK-SAME:       }
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: }

#include "TestHelpers.h"
#include "openmp/kitomp.h"
#include "timer/timer.h"

#include <stdlib.h>

CTOR(RT_TIMER | RT_OPENMP)

static void thrdFn(uint64_t start, uint64_t end, void *args) {
  KitTimerEpoch *e = __kittimer_start("thrd", __kitomp_thread_id());
  __kittimer_stop(e);
}

int main(int argc, char *argv[]) {
  KitTimerEpoch *e = __kittimer_start("main", /*thread=*/0);
  __kitomp_launch(thrdFn, /*beg=*/0, /*end=*/3, /*args=*/NULL, /*argSize=*/0);
  __kittimer_stop(e);

  return 0;
}
