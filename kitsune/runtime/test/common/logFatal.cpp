// -----------------------------------------------------------------------------
// Fatal error messages are always written out, even if verbose mode has not
// been enabled. This will immediately terminate the program with a
// system-dependent error code.
//
// RUN: not %exe 2>&1 > /dev/null | FileCheck %s
// RUN: env KIT_VERBOSE=1 not %exe 2>&1 > /dev/null | FileCheck %s
//
// -----------------------------------------------------------------------------
// Error messages are always written to stderr.
//
// RUN: not %exe 2> /dev/null | FileCheck %s --check-prefix=EMPTY --allow-empty
//
// -----------------------------------------------------------------------------
//
// CHECK: test: ERROR: Fatal error message
//
// EMPTY-NOT: {{^.+$}}
//
// -----------------------------------------------------------------------------

#include "common/logging.h"
#include "kitrt.h"

__attribute__((constructor)) static void ctor(void) { __kitrt_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitrt_finalize(); }

int main(int argc, char* argv[]) {
  kitrt::fatal("test", "Fatal error message");
  return 0;
}
