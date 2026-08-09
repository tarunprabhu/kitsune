// Check that the KIT_VERBOSE and KITRT_VERBOSE environment variables are
// handled correctly when setting verbose mode. Specifically, if both variables
// are set in the environment, the value of KIT_VERBOSE will be used.
//
// RUN: %exe 2>&1 | FileCheck %s --check-prefix=NOT-VERBOSE --allow-empty
//
// RUN: env KIT_VERBOSE=0 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=NOT-VERBOSE --allow-empty
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s --check-prefix=VERBOSE
//
// RUN: env KITRT_VERBOSE=0 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=NOT-VERBOSE --allow-empty
// RUN: env KITRT_VERBOSE=1 %exe 2>&1 | FileCheck %s --check-prefix=VERBOSE
//
// RUN: env KIT_VERBOSE=0 KITRT_VERBOSE=0 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=NOT-VERBOSE --allow-empty
//
// RUN: env KIT_VERBOSE=0 KITRT_VERBOSE=1 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=NOT-VERBOSE --allow-empty
//
// RUN: env KIT_VERBOSE=1 KITRT_VERBOSE=0 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=VERBOSE
//
// RUN: env KIT_VERBOSE=1 KITRT_VERBOSE=1 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=VERBOSE
//
// VERBOSE: Initialized Kitsune runtime (common)
// NOT-VERBOSE-NOT: {{^.+$}}

#include "TestHelpers.h"

CTOR(RT_NONE)

int main(int argc, char *argv[]) { return 0; }
