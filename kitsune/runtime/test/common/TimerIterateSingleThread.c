// Check that the output of timers started and stopped multiple times is as
// expected..
//
// RUN: %exe 2>&1 | FileCheck %s
//
// CHECK: {
// CHECK-NEXT: "pa": {
// CHECK-NEXT: "0": [{{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}]
// CHECK-NEXT: },
// CHECK-NEXT: "papageno": {
// CHECK-NEXT: "0": [{{[0-9]+}}]
// CHECK-NEXT: }
// CHECK-NEXT: }

#include "common/timer.h"
#include "kitrt.h"

__attribute__((constructor)) static void ctor(void) { __kitrt_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitrt_finalize(); }

int main(int argc, char *argv[]) {
  __kittimer_start(11, 0, "papageno");
  for (unsigned i = 0; i < 3; ++i) {
    __kittimer_start(9, 0, "pa");
    __kittimer_stop(9, 0);
  }
  __kittimer_stop(11, 0);

  return 0;
}
