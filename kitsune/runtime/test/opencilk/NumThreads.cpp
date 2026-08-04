// Check that the number of workers used by Kitsune's OpenCilk runtime can be
// be controlled by both KIT_NUM_THREADS and CILK_NWORKERS.
//
// NOTE: With OpenCilk, we cannot set the number of workers to be greater than
// the number of detected CPU's. Since it is difficult to determine the number
// of CPU's available in a truly platform independent way in this test, we just
// set everything to 1.
//
// -----------------------------------------------------------------------------
// RUN: env %exe | FileCheck %s --check-prefix=DEFAULT
//
// DEFAULT: Number of workers = {{[0-9]+}}
//
// -----------------------------------------------------------------------------
// RUN: env KIT_NUM_THREADS=1 %exe | FileCheck %s --check-prefix=KIT
//
// KIT: Number of workers = 1
//
// -----------------------------------------------------------------------------
// RUN: env CILK_NWORKERS=1 %exe | FileCheck %s --check-prefix=WORKERS
//
// WORKERS: Number of workers = 1
//
// -----------------------------------------------------------------------------
// If both KIT_NUM_THREADS and CILK_NWORKERS are set, the former takes
// precedence.
//
// RUN: env KIT_NUM_THREADS=1 CILK_NWORKERS=65537 %exe \
// RUN:     | FileCheck %s --check-prefix=BOTH
//
// BOTH: Number of workers = 1
//
// -----------------------------------------------------------------------------
// If the number of workers being set is greater than the number of available
// CPU's, we fail with an error rather than silently ignoring the value.
//
// RUN: env KIT_NUM_THREADS=65537 not %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=ERROR
//
// RUN: env CILK_NWORKERS=65537 not %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=ERROR
//
// RUN: env KIT_NUM_THREADS=65537 CILK_NWORKERS=65537 not %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=ERROR
//
// ERROR: Number of threads/workers (65537) cannot be greater than number of
// detected CPUs
//
// -----------------------------------------------------------------------------

#include "opencilk/kitocilk.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitocilk_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitocilk_finalize(); }

int main(int argc, char *argv[]) {
  printf("Number of workers = %ld\n", __kitocilk_num_workers());
  return 0;
}
