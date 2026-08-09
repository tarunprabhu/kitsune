// -----------------------------------------------------------------------------
// Fatal error messages are always written out, even if verbose mode has not
// been enabled. This will immediately terminate the program with a
// system-dependent error code.
//
// RUN: not %exe 2>&1 > /dev/null | FileCheck %s --match-full-lines
// RUN: env KIT_VERBOSE=1 not %exe 2>&1 > /dev/null \
// RUN:     | FileCheck %s --match-full-lines
//
// -----------------------------------------------------------------------------
// Error messages are always written to stderr.
//
// RUN: not %exe 2> /dev/null | FileCheck %s --check-prefix=EMPTY --allow-empty
// RUN: not %exe 2>&1 > /dev/null | FileCheck %s --match-full-lines
//
// -----------------------------------------------------------------------------
//
// EMPTY-NOT: {{^.+$}}
//
// CHECK: kitrt: [test]: error: Fatal error message
//
// -----------------------------------------------------------------------------

// This must be defined before common/logging.h is included. In the actual kitrt
// source, this will have been defined by the compiler invocation in the form of
// a `-DKITRT_LOG_TAG="<...>"` command-line option.
#define KITRT_LOG_TAG "test"

#include "TestHelpers.h"
#include "common/logging.h"

CTOR(RT_NONE)

int main(int argc, char *argv[]) {
  FATAL("Fatal error message");
  return 0;
}
