// Check that initializing the runtime multiple times behaves as expected.
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initializing Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (common)
// CHECK: Initializing Kitsune runtime (openmp)
// CHECK: Initialized Kitsune runtime (openmp)
// CHECK: Initializing Kitsune runtime (pthreads)
// CHECK: Initialized Kitsune runtime (pthreads)
// CHECK: Initializing Kitsune runtime (timer)
// CHECK: Initialized Kitsune runtime (timer)
// CHECK: Kitsune runtime already initialized (common)
// CHECK: Kitsune runtime already initialized (openmp)
// CHECK: Kitsune runtime already initialized (pthreads)
// CHECK: Kitsune runtime already initialized (timer)
// CHECK-NOT: Initializing Kitsune runtime (common)
// CHECK-NOT: Initializing Kitsune runtime (openmp)
// CHECK-NOT: Initializing Kitsune runtime (pthreads)
// CHECK-NOT: Initializing Kitsune runtime (timer)

#include "TestHelpers.h"

// It is sufficient to write just one test. Since all runtimes are handled the
// same way, there is no need to check each runtime separately. But, we make
// things a bit more complicated by including several runtimes at once. We are
// careful to pick those that are guaranteed to have been built.
const KitRTInitOptions initOpts{RT_OPENMP | RT_PTHREADS | RT_TIMER};

__attribute__((constructor)) static void ctor(void) {
  __kitrt_initialize(&initOpts);
  __kitrt_initialize(&initOpts);
}

__attribute__((destructor)) static void dtor(void) {
  __kitrt_finalize(&initOpts);
}

MAIN
