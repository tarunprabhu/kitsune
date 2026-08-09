// Check that the number of threads used by Kitsune's pthreads runtime can be
// controlled by KIT_NUM_THREADS.
//
// NOTE: We intentionally use large values here because we would like to set the
// number of threads to a value larger than the number of CPU's that are likely
// to be detected.
//
// -----------------------------------------------------------------------------
// RUN: env %exe | FileCheck %s --check-prefix=DEFAULT
//
// DEFAULT: Number of threads = [[N:[0-9]+]]
//
// -----------------------------------------------------------------------------
// RUN: env KIT_NUM_THREADS=4097 %exe | FileCheck %s --check-prefix=KIT
//
// KIT: Number of threads = 4097
//
// -----------------------------------------------------------------------------

#include "TestHelpers.h"
#include "pthreads/kitpthr.h"

#include <stdio.h>

CTOR(RT_PTHREADS)

int main(int argc, char *argv[]) {
  printf("Number of threads = %ld\n", __kitpthr_num_threads());
  return 0;
}
