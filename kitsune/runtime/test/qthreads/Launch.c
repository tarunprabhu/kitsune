// Check that __kitqthr_launch works as expected.
//
// RUN: KIT_NUM_THREADS=3 %exe | FileCheck %s
//
// CHECK-COUNT-3: From thread

#include "qthreads/kitqthr.h"

#include <pthread.h>
#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitqthr_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitqthr_finalize(); }

static pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;

static void thrdFunc(uint64_t start, uint64_t stop, void *args) {
  pthread_mutex_lock(&mut);
  printf("From thread\n");
  pthread_mutex_unlock(&mut);
}

int main(int argc, char* argv[]) {
  __kitqthr_launch(thrdFunc, 0, 3, NULL);

  pthread_mutex_destroy(&mut);
  return 0;
}
