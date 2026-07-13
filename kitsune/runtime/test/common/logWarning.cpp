// -----------------------------------------------------------------------------
// Warning messages are always written out, even if verbose mode has not been
// enabled.
//
// RUN: %exe 2>&1 > /dev/null | FileCheck %s
// RUN: env KIT_VERBOSE=1 %exe 2>&1 > /dev/null | FileCheck %s
//
// -----------------------------------------------------------------------------
// Error messages are always written to stderr.
//
// RUN: %exe 2> /dev/null | FileCheck %s --check-prefix=EMPTY --allow-empty
//
// -----------------------------------------------------------------------------
//
// CHECK: test: WARNING: Warning message
//
// EMPTY-NOT: {{^.+$}}
//
// -----------------------------------------------------------------------------

#include "common/logging.h"
#include "kitrt.h"

__attribute__((constructor)) static void ctor(void) { __kitrt_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitrt_finalize(); }

int main(int argc, char *argv[]) {
  kitrt::warn("test", "Warning message");
  return 0;
}
