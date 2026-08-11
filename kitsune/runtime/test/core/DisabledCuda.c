// REQUIRES: kitsune-no-cuda
//
// Check that the runtime fails if the cuda runtime is requested when it has not
// been built.
//
// RUN: not %exe 2>&1 | FileCheck %s
//
// CHECK: Kitsune runtime has not been enabled (cuda)

#include "TestHelpers.h"

CTOR(RT_CUDA)

MAIN
