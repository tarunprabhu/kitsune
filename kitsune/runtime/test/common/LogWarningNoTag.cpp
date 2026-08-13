// -----------------------------------------------------------------------------
// Warning messages are always written out, even if verbose mode has not been
// enabled.
//
// RUN: %exe 2>&1 > /dev/null | FileCheck %s --match-full-lines
// RUN: env KIT_VERBOSE=1 %exe 2>&1 > /dev/null \
// RUN:     | FileCheck %s --match-full-lines
//
// -----------------------------------------------------------------------------
// Error messages are always written to stderr.
//
// RUN: %exe 2> /dev/null | FileCheck %s --check-prefix=EMPTY --allow-empty
// RUN: %exe 2>&1 > /dev/null | FileCheck %s --match-full-lines
//
// -----------------------------------------------------------------------------
//
// EMPTY-NOT: {{^.+$}}
//
// CHECK: kitrt: warning: Warning message
//
// -----------------------------------------------------------------------------

#include "TestHelpers.h"
#include "common/logging.h"

CTOR(RT_COMMON)

int main(int argc, char *argv[]) {
  WARN("Warning message");
  return 0;
}
