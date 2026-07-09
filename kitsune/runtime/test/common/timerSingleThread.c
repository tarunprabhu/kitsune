// Check that the output of multiple timers on a single-threaded application is
// as expected.
//
// RUN: %exe | FileCheck %s
//
// CHECK: {
// CHECK-NEXT: "abc": {
// CHECK-NEXT: "0": [{{[0-9]+}}]
// CHECK-NEXT: },
// CHECK-NEXT: "pqr": {
// CHECK-NEXT: "0": [{{[0-9]+}}]
// CHECK-NEXT: },
// CHECK-NEXT: "xyz": {
// CHECK-NEXT: "0": [{{[0-9]+}}]
// CHECK-NEXT: }
// CHECK-NEXT: }

#include "common/timer.h"
#include "kitrt.h"

__attribute__((constructor)) static void ctor(void) { __kitrt_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitrt_finalize(); }

int main(int argc, char *argv[]) {
  __kittimer_start(11, 0, "xyz");
  __kittimer_start(0, 0, "pqr");
  __kittimer_stop(0, 0);
  __kittimer_start(4, 0, "abc");
  __kittimer_stop(4, 0);
  __kittimer_stop(11, 0);

  return 0;
}
