// Check that the number of threads used by the Kitsune's OpenMP runtime can be
// controlled by both KIT_NUM_THREADS and OMP_NUM_THREADS
//
// -----------------------------------------------------------------------------
// RUN: env KIT_VERBOSE=1 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=DEFAULT
//
// DEFAULT: Number of threads = {{[0-9]+}}
//
// -----------------------------------------------------------------------------
// RUN: env KIT_VERBOSE=1 KIT_NUM_THREADS=41 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=KIT
//
// KIT: Number of threads = 41
//
// -----------------------------------------------------------------------------
// RUN: env KIT_VERBOSE=1 OMP_NUM_THREADS=97 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=OMP
//
// OMP: Number of threads = 97
//
// -----------------------------------------------------------------------------
// If both KIT_NUM_THREADS and OMP_NUM_THREADS are set, the former takes
// precedence.
//
// RUN: env KIT_VERBOSE=1 KIT_NUM_THREADS=23 OMP_NUM_THREADS=67 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=BOTH
//
// BOTH: Number of threads = 23
//
// -----------------------------------------------------------------------------

#include "openmp/kitomp.h"

int main(int argc, char* argv[]) {
  __kitomp_initialize();
  __kitomp_finalize();
  return 0;
}
