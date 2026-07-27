// Check that the output of per-thread timers started and stopped multiple times
// is as expected. We use Kitsune's OpenMP runtime because it is guaranteed to
// be built. Also, omp_get_thread_num() returns an integer in
// [0, KIT_NUM_THREADS), so we can match against thread ids.
//
// RUN: env KIT_NUM_THREADS=3 %exe 2>&1 | FileCheck %s
//
// CHECK:      {
// CHECK-NEXT:   "fork": {
// CHECK-NEXT:     "0": [
// CHECK-NEXT:       {{[0-9]+}}
// CHECK-NEXT:     ]
// CHECK-NEXT:   },
// CHECK-NEXT:   "tine": {
// CHECK-NEXT:     "0": [
// CHECK-NEXT:       {{[0-9]+}},
// CHECK-NEXT:       {{[0-9]+}},
// CHECK-NEXT:       {{[0-9]+}}
// CHECK-NEXT:     ]
// CHECK-NEXT:     "1": [
// CHECK-NEXT:       {{[0-9]+}},
// CHECK-NEXT:       {{[0-9]+}},
// CHECK-NEXT:       {{[0-9]+}}
// CHECK-NEXT:     ]
// CHECK-NEXT:     "2": [
// CHECK-NEXT:       {{[0-9]+}},
// CHECK-NEXT:       {{[0-9]+}},
// CHECK-NEXT:       {{[0-9]+}}
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: }

#include "common/timer.h"
#include "openmp/kitomp.h"

#include <stdlib.h>

unsigned omp_get_thread_num(void);

__attribute__((constructor)) static void ctor(void) { __kitomp_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitomp_finalize(); }

static void thrdFn(uint64_t start, uint64_t stop, void *args) {
  KitTimerEpoch *e = __kittimer_start("tine", omp_get_thread_num());
  __kittimer_stop(e);
}

int main(int argc, char *argv[]) {
  KitTimerEpoch *e = __kittimer_start("fork", /*thread=*/0);
  for (unsigned i = 0; i < 3; ++i)
    __kitomp_launch(thrdFn, /*beg=*/0, /*end=*/3, /*args=*/NULL, /*argSize=*/0);
  __kittimer_stop(e);

  return 0;
}
