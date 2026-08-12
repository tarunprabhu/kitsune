// Check that the output of timers started and stopped multiple times is as
// expected.
//
// RUN: %exe 2>&1 | FileCheck %s --check-prefix=DEFAULT
//
// RUN: env KIT_INSTR_SEPARATE=1 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=SEPARATE
//
// DEFAULT:      {
// DEFAULT-NEXT:   "kinder": {
// DEFAULT-NEXT:     "0": [
// DEFAULT-NEXT:       {
// DEFAULT-SAME:         "total": {{[0-9]+}},
// DEFAULT-SAME:         "visits": 3,
// DEFAULT-SAME:         "min": {{[0-9]+}},
// DEFAULT-SAME:         "mean": {{[0-9]+([.][0-9]+)?}},
// DEFAULT-SAME:         "max": {{[0-9]+}}
// DEFAULT-SAME:       }
// DEFAULT-NEXT:     ]
// DEFAULT-NEXT:   },
// DEFAULT-NEXT:   "papageno": {
// DEFAULT-NEXT:     "0": [
// DEFAULT-NEXT:       {
// DEFAULT-SAME:         "total": [[TOT:[0-9]+]],
// DEFAULT-SAME:         "visits": 1,
// DEFAULT-SAME:         "min": [[TOT]],
// DEFAULT-SAME:         "mean": [[TOT]],
// DEFAULT-SAME:         "max": [[TOT]]
// DEFAULT-SAME:       }
// DEFAULT-NEXT:     ]
// DEFAULT-NEXT:   }
// DEFAULT-NEXT: }
//
// SEPARATE:      {
// SEPARATE-NEXT:   "kinder": {
// SEPARATE-NEXT:     "0": [
// SEPARATE-NEXT:       {
// SEPARATE-SAME:         "total": [[K0:[0-9]+]],
// SEPARATE-SAME:         "visits": 1,
// SEPARATE-SAME:         "min": [[K0]],
// SEPARATE-SAME:         "mean": [[K0]],
// SEPARATE-SAME:         "max": [[K0]]
// SEPARATE-SAME:       },
// SEPARATE-NEXT:       {
// SEPARATE-SAME:         "total": [[K1:[0-9]+]],
// SEPARATE-SAME:         "visits": 1,
// SEPARATE-SAME:         "min": [[K1]],
// SEPARATE-SAME:         "mean": [[K1]],
// SEPARATE-SAME:         "max": [[K1]]
// SEPARATE-SAME:       },
// SEPARATE-NEXT:       {
// SEPARATE-SAME:         "total": [[K2:[0-9]+]],
// SEPARATE-SAME:         "visits": 1,
// SEPARATE-SAME:         "min": [[K2]],
// SEPARATE-SAME:         "mean": [[K2]],
// SEPARATE-SAME:         "max": [[K2]]
// SEPARATE-SAME:       }
// SEPARATE-NEXT:     ]
// SEPARATE-NEXT:   },
// SEPARATE-NEXT:   "papageno": {
// SEPARATE-NEXT:     "0": [
// SEPARATE-NEXT:       {
// SEPARATE-SAME:         "total": [[P0:[0-9]+]],
// SEPARATE-SAME:         "visits": 1,
// SEPARATE-SAME:         "min": [[P0]],
// SEPARATE-SAME:         "mean": [[P0]],
// SEPARATE-SAME:         "max": [[P0]]
// SEPARATE-SAME:       }
// SEPARATE-NEXT:     ]
// SEPARATE-NEXT:   }
// SEPARATE-NEXT: }

#include "TestHelpers.h"
#include "timer/timer.h"

CTOR(RT_TIMER)

int main(int argc, char *argv[]) {
  KitTimerEpoch *eo = __kittimer_start("papageno", /*thread=*/0);
  for (unsigned i = 0; i < 3; ++i) {
    KitTimerEpoch *ei = __kittimer_start("kinder", /*thread=*/0);
    __kittimer_stop(ei);
  }
  __kittimer_stop(eo);

  return 0;
}
