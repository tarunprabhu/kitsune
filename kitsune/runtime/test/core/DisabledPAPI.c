// REQUIRES: kitsune-no-papi
//
// Check that the runtime fails if the PAPI support runtime is requested when it
// has not been built.
//
// RUN: not %exe 2>&1 | FileCheck %s
//
// CHECK: Kitsune runtime has not been enabled (papi)

#include "TestHelpers.h"

CTOR(RT_PAPI)

MAIN
