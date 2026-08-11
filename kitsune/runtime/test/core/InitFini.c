// Check that the order in which the constituent runtimes are initialized and
// finalized is as expected. The common runtime should be initialized first,
// followed by the tapir-target-specific runtimes, and finally the support
// runtimes. The order in which the tapir-target runtimes are initialized is
// determined by the numerical value of their RTID's. This is completely
// arbitrary, and may change in the future. In general, the only dependencies
// here are between the support runtimes and the tapir-target-specific ones.
// Within each of these categories, there should be no dependencies.
//
// We do not finalize in strictly reverse order, though we probably should.
// But the three broad categories are finalized in reverse order, so support
// runtimes first, followed by the tapir-target-specific ones, and then the
// common.
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
// CHECK: Finalizing Kitsune runtime (timer)
// CHECK: Finalized Kitsune runtime (timer)
// CHECK: Finalizing Kitsune runtime (openmp)
// CHECK: Finalized Kitsune runtime (openmp)
// CHECK: Finalizing Kitsune runtime (pthreads)
// CHECK: Finalized Kitsune runtime (pthreads)
// CHECK: Finalizing Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (common)

#include "TestHelpers.h"

CTOR(RT_TIMER | RT_PTHREADS | RT_OPENMP)

MAIN
