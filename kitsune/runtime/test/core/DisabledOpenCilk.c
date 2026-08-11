// REQUIRES: kitsune-no-opencilk
//
// Check that the runtime fails if the opencilk runtime is requested when it has
// not been built.
//
// RUN: not %exe 2>&1 | FileCheck %s
//
// CHECK: Kitsune runtime has not been enabled (opencilk)

#include "TestHelpers.h"

CTOR(RT_OPENCILK)

MAIN
