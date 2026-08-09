// By default, times are written to stderr, not stdout.
//
// RUN: %exe 2>&1 > /dev/null | FileCheck %s --check-prefix=STDERR
// RUN: %exe 2> /dev/null | FileCheck %s --check-prefix=STDOUT --allow-empty
//
// STDERR:      {
// STDERR-NEXT:   "sarastro": {
// STDERR-NEXT:     "0": [
// STDERR-NEXT:       {{[0-9]+}}
// STDERR-NEXT:     ]
// STDERR-NEXT:   }
// STDERR-NEXT: }
// STDERR-NOT: {{^.+$}}
//
// STDOUT-NOT: {{^.+$}}

#include "TestHelpers.h"
#include "timer/timer.h"

CTOR(RT_TIMER)

int main(int argc, char *argv[]) {
  kitrt::KitTimerEpoch *e = __kittimer_start("sarastro", /*thread=*/0);
  __kittimer_stop(e);

  return 0;
}
