// Check that the output of per-thread timers started and stopped multiple times
// is as expected. We use Kitsune's OpenMP runtime because it is guaranteed to
// be built. Also, omp_get_thread_num() returns an integer in
// [0, KIT_NUM_THREADS), so one can reasonably collect times across iterations.
//
// RUN: env KIT_NUM_THREADS=3 %exe 2>&1 | FileCheck %s
//
// CHECK: {
// CHECK-NEXT: "fork": {
// CHECK-NEXT: "0": [{{[0-9]+}}]
// CHECK-NEXT: },
// CHECK-NEXT: "tine": {
// CHECK-DAG: "0": [{{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}]
// CHECK-DAG: "1": [{{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}]
// CHECK-DAG: "2": [{{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}]
// CHECK-NEXT: }
// CHECK-NEXT: }

#include "common/timer.h"
#include "openmp/kitomp.h"

#include <stdlib.h>

unsigned omp_get_thread_num(void);

__attribute__((constructor)) static void ctor(void) { __kitomp_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitomp_finalize(); }

static void thrdFn(uint64_t start, uint64_t stop, void *args) {
  TimePoint tick = __kittimer_start();
  __kittimer_stop(tick, 92, omp_get_thread_num(), "tine");
}

int main(int argc, char *argv[]) {
  TimePoint tick = __kittimer_start();
  for (unsigned i = 0; i < 3; ++i)
    __kitomp_launch(thrdFn, 0, 3, NULL);
  __kittimer_stop(tick, 11, 0, "fork");

  return 0;
}
