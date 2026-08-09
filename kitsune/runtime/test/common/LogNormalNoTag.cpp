// -----------------------------------------------------------------------------
// Log messages are not printed if verbose mode is not enabled.
//
// RUN: %exe 2>&1 | FileCheck %s --check-prefix=EMPTY --allow-empty
// RUN: env KIT_VERBOSE=1 %exe 2>&1 \
// RUN:     | FileCheck %s --match-full-lines
// RUN: env KITRT_VERBOSE=1 %exe 2>&1 \
// RUN:     | FileCheck %s --match-full-lines
//
// -----------------------------------------------------------------------------
// Log messages should be written to stderr.
//
// RUN: env KIT_VERBOSE=1 %exe 2> /dev/null \
// RUN:     | FileCheck %s --check-prefix=EMPTY --allow-empty
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 > /dev/null \
// RUN:     | FileCheck %s --match-full-lines
//
// -----------------------------------------------------------------------------
//
// EMPTY-NOT: {{^.+$}}
//
// CHECK: kitrt: Log message
//
// -----------------------------------------------------------------------------

#include "TestHelpers.h"
#include "common/logging.h"

CTOR(RT_NONE)

int main(int argc, char *argv[]) {
  LOG("Log message");
  return 0;
}
