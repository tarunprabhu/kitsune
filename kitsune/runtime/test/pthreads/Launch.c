// Check that __kitpthr_async_launch works as expected.
//
// RUN: KIT_NUM_THREADS=3 %exe | FileCheck %s
//
// CHECK-COUNT-3: From thread

#include "pthreads/kitpthr.h"

#include <pthread.h>
#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitpthr_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitpthr_finalize(); }

static pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;

static void thrdFunc(uint64_t start, uint64_t stop, void *args) {
  pthread_mutex_lock(&mut);
  printf("From thread\n");
  pthread_mutex_unlock(&mut);
}

int main(int argc, char *argv[]) {
  KitPthrLaunchContext *ctx =
      __kitpthr_async_launch(thrdFunc, /*beg=*/0, /*end=*/3,
                             /*args=*/NULL, /*argSize=*/0);
  __kitpthr_sync(ctx);

  pthread_mutex_destroy(&mut);
  return 0;
}
