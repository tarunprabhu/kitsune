// REQUIRES: kitsune-papi
//
// If no events are measured, nothing should be printed at runtime.
//
// RUN: %exe 2>&1 | FileCheck %s --allow-empty
//
// CHECK-NOT: {{^.+$}}

#include "TestHelpers.h"

CTOR(RT_PAPI)

MAIN
