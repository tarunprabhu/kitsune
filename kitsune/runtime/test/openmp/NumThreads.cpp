// Check that the number of threads used by Kitsune's OpenMP runtime can be
// controlled by both KIT_NUM_THREADS and OMP_NUM_THREADS.
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
// RUN: env OMP_NUM_THREADS=4095 %exe | FileCheck %s --check-prefix=OMP
//
// OMP: Number of threads = 4095
//
// -----------------------------------------------------------------------------
// If both KIT_NUM_THREADS and OMP_NUM_THREADS are set, the former takes
// precedence.
//
// RUN: env KIT_NUM_THREADS=4097 OMP_NUM_THREADS=4095 %exe \
// RUN:     | FileCheck %s --check-prefix=BOTH
//
// BOTH: Number of threads = 4097
//
// -----------------------------------------------------------------------------

#include "TestHelpers.h"
#include "openmp/kitomp.h"

#include <stdio.h>

CTOR(RT_OPENMP)

int main(int argc, char *argv[]) {
  printf("Number of threads = %ld\n", __kitomp_num_threads());
  return 0;
}
