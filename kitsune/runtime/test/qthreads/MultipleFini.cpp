// Check that finalizing the runtime multiple times has no effect. The runtime
// should remain finalized.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initialized = {{[1-9][0-9]*}}
// CHECK: Finalizing Kitsune runtime (qthreads)
// CHECK: Finalized Kitsune runtime (qthreads)
// CHECK: Finalizing Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (common)
// CHECK: Initialized = 0
// CHECK: Cannot finalize runtime. Not initialized
// CHECK: Initialized = 0

#include <qthreads/kitqthr.h>

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitqthr_initialize(); }

__attribute__((destructor)) static void dtor(void) {
  fprintf(stderr, "Initialized = %d\n", __kitqthr_initialized());
  __kitqthr_finalize();
  fprintf(stderr, "Initialized = %d\n", __kitqthr_initialized());
  __kitqthr_finalize();
  fprintf(stderr, "Initialized = %d\n", __kitqthr_initialized());
}

int main(int argc, char *argv[]) { return 0; }
