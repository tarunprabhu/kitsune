// Check that the output of per-thread timers started and stopped multiple times
// is as expected. We use Kitsune's OpenMP runtime because it is guaranteed to
// be built. Also, omp_get_thread_num() returns an integer in
// [0, KIT_NUM_THREADS), so we can match against thread ids.
//
// RUN: env KIT_NUM_THREADS=3 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=DEFAULT
//
// RUN: env KIT_NUM_THREADS=3 KIT_INSTR_SEPARATE=1 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=SEPARATE
//
// DEFAULT:      {
// DEFAULT-NEXT:   "fork": {
// DEFAULT-NEXT:     "0": [
// DEFAULT-NEXT:       {{[0-9]+}}
// DEFAULT-NEXT:     ]
// DEFAULT-NEXT:   },
// DEFAULT-NEXT:   "tine": {
// DEFAULT-NEXT:     "0": [
// DEFAULT-NEXT:       {{[0-9]+}}
// DEFAULT-NEXT:     ]
// DEFAULT-NEXT:     "1": [
// DEFAULT-NEXT:       {{[0-9]+}}
// DEFAULT-NEXT:     ]
// DEFAULT-NEXT:     "2": [
// DEFAULT-NEXT:       {{[0-9]+}}
// DEFAULT-NEXT:     ]
// DEFAULT-NEXT:   }
// DEFAULT-NEXT: }
//
// SEPARATE:      {
// SEPARATE-NEXT:   "fork": {
// SEPARATE-NEXT:     "0": [
// SEPARATE-NEXT:       {{[0-9]+}}
// SEPARATE-NEXT:     ]
// SEPARATE-NEXT:   },
// SEPARATE-NEXT:   "tine": {
// SEPARATE-NEXT:     "0": [
// SEPARATE-NEXT:       {{[0-9]+}},
// SEPARATE-NEXT:       {{[0-9]+}},
// SEPARATE-NEXT:       {{[0-9]+}}
// SEPARATE-NEXT:     ]
// SEPARATE-NEXT:     "1": [
// SEPARATE-NEXT:       {{[0-9]+}},
// SEPARATE-NEXT:       {{[0-9]+}},
// SEPARATE-NEXT:       {{[0-9]+}}
// SEPARATE-NEXT:     ]
// SEPARATE-NEXT:     "2": [
// SEPARATE-NEXT:       {{[0-9]+}},
// SEPARATE-NEXT:       {{[0-9]+}},
// SEPARATE-NEXT:       {{[0-9]+}}
// SEPARATE-NEXT:     ]
// SEPARATE-NEXT:   }
// SEPARATE-NEXT: }

#include "TestHelpers.h"
#include "openmp/kitomp.h"
#include "timer/timer.h"

#include <stdlib.h>

CTOR(RT_TIMER | RT_OPENMP)

static void thrdFn(uint64_t start, uint64_t stop, void *args) {
  kitrt::KitTimerEpoch *e = __kittimer_start("tine", __kitomp_thread_id());
  __kittimer_stop(e);
}

int main(int argc, char *argv[]) {
  kitrt::KitTimerEpoch *e = __kittimer_start("fork", /*thread=*/0);
  for (unsigned i = 0; i < 3; ++i)
    __kitomp_launch(thrdFn, /*beg=*/0, /*end=*/3, /*args=*/NULL, /*argSize=*/0);
  __kittimer_stop(e);

  return 0;
}
