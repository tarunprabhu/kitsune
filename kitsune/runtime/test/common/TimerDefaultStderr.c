// By default, times are written to stderr, not stdout.
//
// RUN: %exe 2>&1 > /dev/null | FileCheck %s --check-prefix=STDERR
// RUN: %exe 2> /dev/null | FileCheck %s --check-prefix=STDOUT --allow-empty
//
// STDERR: {
// STDERR-NEXT: "sarastro": {
// STDERR-NEXT: "0": [{{[0-9]+}}]
// STDERR-NEXT: }
// STDERR-NEXT: }
// STDERR-NOT: {{^.+$}}
//
// STDOUT-NOT: {{^.+$}}

#include "kitrt.h"

__attribute__((constructor)) static void ctor(void) { __kitrt_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitrt_finalize(); }

int main(int argc, char *argv[]) {
  TimePoint tick = __kittimer_start();
  __kittimer_stop(tick, /*timerID=*/11, /*threadID=*/0, "sarastro");

  return 0;
}
