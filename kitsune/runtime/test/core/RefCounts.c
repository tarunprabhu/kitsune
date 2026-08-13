// Check that runtimes are only finalized when the number of finalize calls
// matches the number of initialize calls.
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initializing Kitsune runtime (common)
// CHECK: Initializing Kitsune runtime (openmp)
// CHECK: Initializing Kitsune runtime (pthreads)
// CHECK: Kitsune runtime already initialized (common)
// CHECK: Kitsune runtime already initialized (openmp)
// CHECK: Initializing Kitsune runtime (timer)
// CHECK: Kitsune runtime already initialized (common)
// CHECK: Finalizing Kitsune runtime (timer)
// CHECK: Not finalizing Kitsune runtime. Uses remain (common)
// CHECK: Not finalizing Kitsune runtime. Uses remain (common)
// CHECK: Finalizing Kitsune runtime (pthreads)
// CHECK: Finalizing Kitsune runtime (openmp)
// CHECK: Finalizing Kitsune runtime (common)

#include "TestHelpers.h"

KitRTInitOptions conf1 = {RT_OPENMP | RT_PTHREADS};
KitRTInitOptions conf2 = {RT_OPENMP | RT_TIMER};
KitRTInitOptions conf3 = {RT_COMMON};

__attribute__((constructor)) static void ctor(void) {
  __kitrt_initialize(&conf1);
  __kitrt_initialize(&conf2);
  __kitrt_initialize(&conf3);
}

__attribute__((destructor)) static void dtor(void) {
  // The order in which __kitrt_finalize is called may not mirror the order in
  // which __kitrt_initialize is called.
  __kitrt_finalize(&conf2);
  __kitrt_finalize(&conf3);
  __kitrt_finalize(&conf1);
}

MAIN
