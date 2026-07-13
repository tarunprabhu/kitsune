// -----------------------------------------------------------------------------
// Error messages are always written out, even if verbose mode has not been
// enabled. But an error will not result in a failure.
//
// RUN: %exe 2>&1 > /dev/null | FileCheck %s
// RUN: KIT_VERBOSE=1 %exe 2>&1 > /dev/null | FileCheck %s
//
// -----------------------------------------------------------------------------
// Error messages are always written to stderr.
//
// RUN: %exe 2> /dev/null | FileCheck %s --check-prefix=EMPTY --allow-empty
// RUN: %exe 2>&1 > /dev/null | FileCheck %s
//
// -----------------------------------------------------------------------------
//
// EMPTY-NOT: {{^.+$}}
//
// CHECK: test: ERROR: Error message
//
// -----------------------------------------------------------------------------

#include "common/logging.h"
#include "kitrt.h"

__attribute__((constructor)) static void ctor(void) { __kitrt_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitrt_finalize(); }

int main(int argc, char *argv[]) {
  kitrt::error("test", "Error message");
  return 0;
}
