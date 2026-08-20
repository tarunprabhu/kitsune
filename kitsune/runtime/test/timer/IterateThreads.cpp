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
// DEFAULT-NEXT:       {
// DEFAULT-SAME:         "total": [[F:[0-9]+]],
// DEFAULT-SAME:         "visits": 1,
// DEFAULT-SAME:         "min": [[F]],
// DEFAULT-SAME:         "mean": [[F]],
// DEFAULT-SAME:         "max": [[F]]
// DEFAULT-SAME:       }
// DEFAULT-NEXT:     ]
// DEFAULT-NEXT:   },
// DEFAULT-NEXT:   "tine": {
// DEFAULT-NEXT:     "0": [
// DEFAULT-NEXT:       {
// DEFAULT-SAME:         "total": {{[0-9]+}},
// DEFAULT-SAME:         "visits": 3,
// DEFAULT-SAME:         "min": {{[0-9]+}},
// DEFAULT-SAME:         "mean": {{[0-9]+([.][0-9]+)?}},
// DEFAULT-SAME:         "max": {{[0-9]+}}
// DEFAULT-SAME:       }
// DEFAULT-NEXT:     ]
// DEFAULT-NEXT:     "1": [
// DEFAULT-NEXT:       {
// DEFAULT-SAME:         "total": {{[0-9]+}},
// DEFAULT-SAME:         "visits": 3,
// DEFAULT-SAME:         "min": {{[0-9]+}},
// DEFAULT-SAME:         "mean": {{[0-9]+([.][0-9]+)?}},
// DEFAULT-SAME:         "max": {{[0-9]+}}
// DEFAULT-SAME:       }
// DEFAULT-NEXT:     ]
// DEFAULT-NEXT:     "2": [
// DEFAULT-NEXT:       {
// DEFAULT-SAME:         "total": {{[0-9]+}},
// DEFAULT-SAME:         "visits": 3,
// DEFAULT-SAME:         "min": {{[0-9]+}},
// DEFAULT-SAME:         "mean": {{[0-9]+([.][0-9]+)?}},
// DEFAULT-SAME:         "max": {{[0-9]+}}
// DEFAULT-SAME:       }
// DEFAULT-NEXT:     ]
// DEFAULT-NEXT:   }
// DEFAULT-NEXT: }
//
// SEPARATE:      {
// SEPARATE-NEXT:   "fork": {
// SEPARATE-NEXT:     "0": [
// SEPARATE-NEXT:       {
// SEPARATE-SAME:         "total": {{[0-9]+}},
// SEPARATE-SAME:         "visits": 1,
// SEPARATE-SAME:         "min": {{[0-9]+}},
// SEPARATE-SAME:         "mean": {{[0-9]+([.][0-9]+)?}},
// SEPARATE-SAME:         "max": {{[0-9]+}}
// SEPARATE-SAME:       }
// SEPARATE-NEXT:     ]
// SEPARATE-NEXT:   },
// SEPARATE-NEXT:   "tine": {
// SEPARATE-NEXT:     "0": [
// SEPARATE-NEXT:       {
// SEPARATE-SAME:         "total": [[T00:[0-9]+]],
// SEPARATE-SAME:         "visits": 1,
// SEPARATE-SAME:         "min": [[T00]],
// SEPARATE-SAME:         "mean": [[T00]],
// SEPARATE-SAME:         "max": [[T00]]
// SEPARATE-SAME:       },
// SEPARATE-NEXT:       {
// SEPARATE-SAME:         "total": [[T01:[0-9]+]],
// SEPARATE-SAME:         "visits": 1,
// SEPARATE-SAME:         "min": [[T01]],
// SEPARATE-SAME:         "mean": [[T01]],
// SEPARATE-SAME:         "max": [[T01]]
// SEPARATE-SAME:       },
// SEPARATE-NEXT:       {
// SEPARATE-SAME:         "total": [[T02:[0-9]+]],
// SEPARATE-SAME:         "visits": 1,
// SEPARATE-SAME:         "min": [[T02]],
// SEPARATE-SAME:         "mean": [[T02]],
// SEPARATE-SAME:         "max": [[T02]]
// SEPARATE-SAME:       }
// SEPARATE-NEXT:     ]
// SEPARATE-NEXT:     "1": [
// SEPARATE-NEXT:       {
// SEPARATE-SAME:         "total": [[T10:[0-9]+]],
// SEPARATE-SAME:         "visits": 1,
// SEPARATE-SAME:         "min": [[T10]],
// SEPARATE-SAME:         "mean": [[T10]],
// SEPARATE-SAME:         "max": [[T10]]
// SEPARATE-SAME:       },
// SEPARATE-NEXT:       {
// SEPARATE-SAME:         "total": [[T11:[0-9]+]],
// SEPARATE-SAME:         "visits": 1,
// SEPARATE-SAME:         "min": [[T11]],
// SEPARATE-SAME:         "mean": [[T11]],
// SEPARATE-SAME:         "max": [[T11]]
// SEPARATE-SAME:       },
// SEPARATE-NEXT:       {
// SEPARATE-SAME:         "total": [[T12:[0-9]+]],
// SEPARATE-SAME:         "visits": 1,
// SEPARATE-SAME:         "min": [[T12]],
// SEPARATE-SAME:         "mean": [[T12]],
// SEPARATE-SAME:         "max": [[T12]]
// SEPARATE-SAME:       }
// SEPARATE-NEXT:     ]
// SEPARATE-NEXT:     "2": [
// SEPARATE-NEXT:       {
// SEPARATE-SAME:         "total": [[T20:[0-9]+]],
// SEPARATE-SAME:         "visits": 1,
// SEPARATE-SAME:         "min": [[T20]],
// SEPARATE-SAME:         "mean": [[T20]],
// SEPARATE-SAME:         "max": [[T20]]
// SEPARATE-SAME:       },
// SEPARATE-NEXT:       {
// SEPARATE-SAME:         "total": [[T21:[0-9]+]],
// SEPARATE-SAME:         "visits": 1,
// SEPARATE-SAME:         "min": [[T21]],
// SEPARATE-SAME:         "mean": [[T21]],
// SEPARATE-SAME:         "max": [[T21]]
// SEPARATE-SAME:       },
// SEPARATE-NEXT:       {
// SEPARATE-SAME:         "total": [[T22:[0-9]+]],
// SEPARATE-SAME:         "visits": 1,
// SEPARATE-SAME:         "min": [[T22]],
// SEPARATE-SAME:         "mean": [[T22]],
// SEPARATE-SAME:         "max": [[T22]]
// SEPARATE-SAME:       }
// SEPARATE-NEXT:     ]
// SEPARATE-NEXT:   }
// SEPARATE-NEXT: }

#include "TestHelpers.h"
#include "openmp/kitomp.h"
#include "timer/timer.h"

#include <stdlib.h>

CTOR(RT_TIMER | RT_OPENMP)

static void thrdFn(uint64_t start, void *args) {
  KitTimerEpoch *e = __kittimer_start("tine", __kitomp_thread_id());
  __kittimer_stop(e);
}

int main(int argc, char *argv[]) {
  KitTimerEpoch *e = __kittimer_start("fork", /*thread=*/0);
  for (unsigned i = 0; i < 3; ++i)
    __kitomp_launch(thrdFn, /*beg=*/0, /*end=*/3, /*args=*/NULL, /*argSize=*/0);
  __kittimer_stop(e);

  return 0;
}
