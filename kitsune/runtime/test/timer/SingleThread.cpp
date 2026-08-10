// Check that the output of multiple timers on a single-threaded application is
// as expected.
//
// RUN: %exe 2>&1 | FileCheck %s
//
// CHECK:      {
// CHECK-NEXT:   "monostatos": {
// CHECK-NEXT:     "0": [
// CHECK-NEXT:       {{[0-9]+}}
// CHECK-NEXT:     ]
// CHECK-NEXT:   },
// CHECK-NEXT:   "pamina": {
// CHECK-NEXT:     "0": [
// CHECK-NEXT:      {{[0-9]+}}
// CHECK-NEXT:     ]
// CHECK-NEXT:   },
// CHECK-NEXT:   "tamino": {
// CHECK-NEXT:     "0": [
// CHECK-NEXT:       {{[0-9]+}}
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: }

#include "TestHelpers.h"
#include "timer/timer.h"

CTOR(RT_TIMER)

int main(int argc, char *argv[]) {
  KitTimerEpoch *e0 = __kittimer_start("tamino", /*thread=*/0);

  KitTimerEpoch *e1 = __kittimer_start("pamina", /*thread=*/0);
  __kittimer_stop(e1);

  KitTimerEpoch *e2 = __kittimer_start("monostatos", /*thread=*/0);
  __kittimer_stop(e2);

  __kittimer_stop(e0);

  return 0;
}
