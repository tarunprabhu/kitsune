// Check that the output of per-thread timers on a multi-threaded application is
// as expected. We use Kitsune's pthreads runtime because it is guaranteed to be
// built.
//
// RUN: env KIT_NUM_THREADS=3 %exe | FileCheck %s
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

static void thrdFn(int64_t start, int64_t end, void *args) {
  __kittimer_start(92, pthread_self(), "thrd");
  __kittimer_stop(92, pthread_self());
}

int main(int argc, char *argv[]) {
  __kittimer_start(11, 0, "main");
  KitPthrContext *ctx = __kitpthr_launch(thrdFn, 0, 3, NULL);
  __kitpthr_sync(ctx);
  __kittimer_stop(11, 0);

  return 0;
}
