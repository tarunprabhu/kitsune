// Check that the output of per-thread timers on a multi-threaded application is
// as expected. We use Kitsune's pthreads runtime because it is guaranteed to be
// built.
//
// RUN: env KIT_NUM_THREADS=3 %exe 2>&1 | FileCheck %s
//
// CHECK: {
// CHECK-NEXT: "main": {
// CHECK-NEXT: "0": [{{[0-9]+}}]
// CHECK-NEXT: },
// CHECK-NEXT: "thrd": {
// CHECK-NEXT: "{{[0-9]+}}": [{{[0-9]+}}],
// CHECK-NEXT: "{{[0-9]+}}": [{{[0-9]+}}],
// CHECK-NEXT: "{{[0-9]+}}": [{{[0-9]+}}]
// CHECK-NEXT: }
// CHECK-NEXT: }

#include "common/timer.h"
#include "pthreads/kitpthr.h"

#include <pthread.h>
#include <stdlib.h>

__attribute__((constructor)) static void ctor(void) { __kitpthr_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitpthr_finalize(); }

static void thrdFn(uint64_t start, uint64_t end, void *args) {
  TimePoint tick = __kittimer_start();
  __kittimer_stop(tick, /*timerID=*/92, /*threadID=*/pthread_self(), "thrd");
}

int main(int argc, char *argv[]) {
  TimePoint tick = __kittimer_start();
  KitPthrLaunchContext *ctx =
      __kitpthr_async_launch(thrdFn, /*beg=*/0, /*end=*/3,
                             /*args=*/NULL, /*argSize=*/0);
  __kitpthr_sync(ctx);
  __kittimer_stop(tick, /*timerID=*/11, /*threadID=*/0, "main");

  return 0;
}
