// Check that the output of timers started and stopped multiple times is as
// expected.
//
// RUN: %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=DEFAULT
//
// RUN: env KIT_INSTR_SEPARATE=1 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=SEPARATE
//
// DEFAULT:      {
// DEFAULT-NEXT:   "kinder": {
// DEFAULT-NEXT:     "0": [
// DEFAULT-NEXT:       {{[0-9]+}}
// DEFAULT-NEXT:     ]
// DEFAULT-NEXT:   },
// DEFAULT-NEXT:   "papageno": {
// DEFAULT-NEXT:     "0": [
// DEFAULT-NEXT:       {{[0-9]+}}
// DEFAULT-NEXT:     ]
// DEFAULT-NEXT:   }
// DEFAULT-NEXT: }
//
// SEPARATE:      {
// SEPARATE-NEXT:   "kinder": {
// SEPARATE-NEXT:     "0": [
// SEPARATE-NEXT:       {{[0-9]+}},
// SEPARATE-NEXT:       {{[0-9]+}},
// SEPARATE-NEXT:       {{[0-9]+}}
// SEPARATE-NEXT:     ]
// SEPARATE-NEXT:   },
// SEPARATE-NEXT:   "papageno": {
// SEPARATE-NEXT:     "0": [
// SEPARATE-NEXT:       {{[0-9]+}}
// SEPARATE-NEXT:     ]
// SEPARATE-NEXT:   }
// SEPARATE-NEXT: }

#include "common/timer.h"
#include "kitrt.h"

__attribute__((constructor)) static void ctor(void) { __kitrt_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitrt_finalize(); }

int main(int argc, char *argv[]) {
  KitTimerEpoch *eo = __kittimer_start("papageno", /*thread=*/0);
  for (unsigned i = 0; i < 3; ++i) {
    KitTimerEpoch *ei = __kittimer_start("kinder", /*thread=*/0);
    __kittimer_stop(ei);
  }
  __kittimer_stop(eo);

  return 0;
}
