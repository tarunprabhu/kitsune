// Check that the output of timers started and stopped multiple times is as
// expected.
//
// RUN: %exe 2>&1 | FileCheck %s
//
// CHECK:      {
// CHECK-NEXT:   "papageno": {
// CHECK-NEXT:     "0": [
// CHECK-NEXT:       {{[0-9]+}}
// CHECK-NEXT:     ]
// CHECK-NEXT:   },
// CHECK-NEXT:   "kinder": {
// CHECK-NEXT:     "0": [
// CHECK-NEXT:       {{[0-9]+}},
// CHECK-NEXT:       {{[0-9]+}},
// CHECK-NEXT:       {{[0-9]+}}
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: }

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
