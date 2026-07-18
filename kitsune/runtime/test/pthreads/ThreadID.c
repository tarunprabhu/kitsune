// Check that __kitpthr_thread_id works as expected.
//
// RUN: KIT_NUM_THREADS=3 %exe | FileCheck %s
//
// CHECK-COUNT-3: {{[0-9]+}}
// CHECK-NOT: {{^.+$}}

#include "pthreads/kitpthr.h"

#include <pthread.h>
#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitpthr_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitpthr_finalize(); }

static pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;

static void thrdFunc(uint64_t start, uint64_t stop, void *args) {
  pthread_mutex_lock(&mut);
  printf("%ld\n", __kitpthr_thread_id());
  pthread_mutex_unlock(&mut);
}

int main(int argc, char *argv[]) {
  KitPthrLaunchContext *ctx = __kitpthr_launch(thrdFunc, 0, 3, NULL);
  __kitpthr_sync(ctx);

  pthread_mutex_destroy(&mut);
  return 0;
}
