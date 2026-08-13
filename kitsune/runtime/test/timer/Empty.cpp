// If nothing is timed, nothing should be printed.
//
// RUN: %exe 2>&1 | FileCheck %s --allow-empty
//
// CHECK-NOT: {{^.+$}}

#include "TestHelpers.h"

CTOR(RT_TIMER)

MAIN
