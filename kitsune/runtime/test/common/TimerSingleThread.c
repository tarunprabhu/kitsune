// Check that the output of multiple timers on a single-threaded application is
// as expected.
//
// RUN: %exe 2>&1 | FileCheck %s
//
// CHECK: {
// CHECK-NEXT: "monostatos": {
// CHECK-NEXT: "0": [{{[0-9]+}}]
// CHECK-NEXT: },
// CHECK-NEXT: "pamina": {
// CHECK-NEXT: "0": [{{[0-9]+}}]
// CHECK-NEXT: },
// CHECK-NEXT: "tamino": {
// CHECK-NEXT: "0": [{{[0-9]+}}]
// CHECK-NEXT: }
// CHECK-NEXT: }

#include "common/timer.h"
#include "kitrt.h"

__attribute__((constructor)) static void ctor(void) { __kitrt_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitrt_finalize(); }

int main(int argc, char *argv[]) {
  TimePoint tick0 = __kittimer_start();

  TimePoint tick1 = __kittimer_start();
  __kittimer_stop(tick1, /*timerID=*/0, /*threadID=*/0, "pamina");

  TimePoint tick2 = __kittimer_start();
  __kittimer_stop(tick2, /*timerID=*/4, /*threadID=*/0, "monostatos");

  __kittimer_stop(tick0, /*timerID=*/11, /*threadID=*/0, "tamino");

  return 0;
}
