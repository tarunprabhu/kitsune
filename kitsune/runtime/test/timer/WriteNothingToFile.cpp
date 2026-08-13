// If nothing is timed, nothing should be written to the specified timing file.
// The timing file should not be created.
//
// RUN: rm -rf %t.json
// RUN: env KIT_TIMING_FILE=%t.json %exe | FileCheck %s --allow-empty
// RUN: not cat %t.json | FileCheck %s --check-prefix=NOEXIST --allow-empty
//
// NOEXIST-NOT: {{^.+$}}
//
// -----------------------------------------------------------------------------
// If the file already exists, it must remain unchanged.
//
// RUN: echo "Contents" > %t.existing.json
// RUN: env KIT_TIMING_FILE=%t.json %exe | FileCheck %s --allow-empty
// RUN: cat %t.existing.json | FileCheck %s --check-prefix=UNCHANGED
//
// UNCHANGED: Contents
// UNCHANGED-NOT: {{^.+$}}
//
// CHECK-NOT: {{^.+$}}
// -----------------------------------------------------------------------------

#include "TestHelpers.h"

CTOR(RT_TIMER)

MAIN
