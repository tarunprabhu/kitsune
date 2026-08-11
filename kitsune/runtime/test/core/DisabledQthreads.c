// REQUIRES: kitsune-no-qthreads
//
// Check that the runtime fails if the qthreads runtime is requested when it has
// not been built.
//
// RUN: not %exe 2>&1 | FileCheck %s
//
// CHECK: Kitsune runtime has not been enabled (qthreads)

#include "TestHelpers.h"

CTOR(RT_QTHREADS)

MAIN
