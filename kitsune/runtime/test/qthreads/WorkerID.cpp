// Check that __kitqthr_worker_id works as expected.
//
// RUN: KIT_NUM_THREADS=3 %exe | FileCheck %s
//
// CHECK-DAG: 1
// CHECK-DAG: 2
// CHECK-DAG: 3
// CHECK-NOT: {{^.+$}}

#include "TestHelpers.h"
#include "qthreads/kitqthr.h"

#include <pthread.h>
#include <stdio.h>

CTOR(RT_QTHREADS)

static pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;

static void thrdFunc(uint64_t start, void *args) {
  pthread_mutex_lock(&mut);
  printf("%ld\n", __kitqthr_worker_id());
  pthread_mutex_unlock(&mut);
}

int main(int argc, char *argv[]) {
  __kitqthr_launch(thrdFunc, /*beg=*/0, /*end=*/3, /*args=*/NULL,
                   /*argSize=*/0);

  pthread_mutex_destroy(&mut);
  return 0;
}
