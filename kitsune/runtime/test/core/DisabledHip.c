// REQUIRES: kitsune-no-hip
//
// Check that the runtime fails if the hip runtime is requested when it has not
// been built.
//
// RUN: not %exe 2>&1 | FileCheck %s
//
// CHECK: Kitsune runtime has not been enabled (hip)

#include "TestHelpers.h"

CTOR(RT_HIP)

MAIN
