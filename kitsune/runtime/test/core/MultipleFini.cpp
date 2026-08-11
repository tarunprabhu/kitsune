// Check that initializing the runtime multiple times behaves as expected.
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Finalizing Kitsune runtime (timer)
// CHECK: Finalized Kitsune runtime (timer)
// CHECK: Finalizing Kitsune runtime (openmp)
// CHECK: Finalized Kitsune runtime (openmp)
// CHECK: Finalizing Kitsune runtime (pthreads)
// CHECK: Finalized Kitsune runtime (pthreads)
// CHECK: Finalizing Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (common)
// CHECK: Cannot finalize Kitsune runtime. Not initialized (timer)
// CHECK: Cannot finalize Kitsune runtime. Not initialized (openmp)
// CHECK: Cannot finalize Kitsune runtime. Not initialized (pthreads)
// CHECK: Cannot finalize Kitsune runtime. Not initialized (common)
// CHECK-NOT: Finalizing Kitsune runtime (timer)
// CHECK-NOT: Finalizing Kitsune runtime (openmp)
// CHECK-NOT: Finalizing Kitsune runtime (pthreads)
// CHECK-NOT: Finalizing Kitsune runtime (common)

#include "TestHelpers.h"

// It is sufficient to write just one test. Since all runtimes are handled the
// same way, there is no need to check each runtime separately. But, we make
// things a bit more complicated by including several runtimes at once. We are
// careful to pick those that are guaranteed to have been built.
const KitRTInitOptions initOpts{RT_OPENMP | RT_PTHREADS | RT_TIMER};

__attribute__((constructor)) static void ctor(void) {
  __kitrt_initialize(&initOpts);
}

__attribute__((destructor)) static void dtor(void) {
  __kitrt_finalize(&initOpts);
  __kitrt_finalize(&initOpts);
}

MAIN
