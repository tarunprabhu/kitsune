// Check that the number of threads used by Kitsune's OpenMP runtime can be
// controlled by both KIT_NUM_THREADS. While qthreads can also be controlled
// using QT_NUM_SHEPHERDS and QT_WORKERS_PER_SHEPHERD, we don't officially
// support this.
//
// -----------------------------------------------------------------------------
// RUN: env %exe | FileCheck %s --check-prefix=DEFAULT
//
// DEFAULT: Number of workers = [[N:[0-9]+]]
// DEFAULT: Number of shepherds = [[N]]
//
// -----------------------------------------------------------------------------
// Unlike some of the other runtimes. we cannot set the number of threads to a
// large number because it causes severe performance degradation - presumably
// because the qthreads runtime launches as many threads as requested shepherds.
// We could try to set KIT_NUM_THREADS to `NUM_CPUS + 1`, but there is no
// platform-independent way of determining this from a script.
//
// RUN: env KIT_NUM_THREADS=7 %exe | FileCheck %s --check-prefix=KIT
//
// KIT: Number of workers = 7
// KIT: Number of shepherds = 7
//
// -----------------------------------------------------------------------------

#include "qthreads/kitqthr.h"

#include <qthread.h>
#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitqthr_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitqthr_finalize(); }

int main(int argc, char *argv[]) {
  printf("Number of workers = %ld\n", __kitqthr_num_workers());

  // To make proper use of the cores available on the system, it seems that
  // qthreads must be set to use as many shepherds as there are cores with one
  // worker per shepherd. Check that we do set these correctly as well.
  printf("Number of shepherds = %d\n", qthread_num_shepherds());

  return 0;
}
