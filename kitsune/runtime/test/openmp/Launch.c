// Check that __kitomp_launch works as expected.
//
// RUN: KIT_NUM_THREADS=3 %exe | FileCheck %s
//
// CHECK-COUNT-3: From thread

#include "openmp/kitomp.h"

#include <pthread.h>
#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitomp_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitomp_finalize(); }

static pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;

static void thrdFunc(uint64_t start, uint64_t stop, void *args) {
  pthread_mutex_lock(&mut);
  printf("From thread\n");
  pthread_mutex_unlock(&mut);
}

int main(int argc, char* argv[]) {
  __kitomp_launch(thrdFunc, 0, 3, NULL);

  pthread_mutex_destroy(&mut);
  return 0;
}
