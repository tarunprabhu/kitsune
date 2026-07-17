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
  __kittimer_start(11, 0, "tamino");
  __kittimer_start(0, 0, "pamina");
  __kittimer_stop(0, 0);
  __kittimer_start(4, 0, "monostatos");
  __kittimer_stop(4, 0);
  __kittimer_stop(11, 0);

  return 0;
}
