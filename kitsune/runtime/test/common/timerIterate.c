// Check that the output of timers started and stopped multiple times is as
// expected..
//
// RUN: %exe | FileCheck %s
//
// CHECK: {
// CHECK-NEXT: "loop": {
// CHECK-NEXT: "0": [{{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}]
// CHECK-NEXT: },
// CHECK-NEXT: "main": {
// CHECK-NEXT: "0": [{{[0-9]+}}]
// CHECK-NEXT: }
// CHECK-NEXT: }

#include "common/timer.h"
#include "kitrt.h"

__attribute__((constructor)) static void ctor(void) { __kitrt_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitrt_finalize(); }

int main(int argc, char *argv[]) {
  __kittimer_start(11, 0, "main");
  for (unsigned i = 0; i < 3; ++i) {
    __kittimer_start(9, 0, "loop");
    __kittimer_stop(9, 0);
  }
  __kittimer_stop(11, 0);

  return 0;
}
