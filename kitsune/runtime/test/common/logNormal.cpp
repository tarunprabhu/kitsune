// -----------------------------------------------------------------------------
// Log messages are not printed if verbose mode is not enabled.
//
// RUN: %exe 2>&1 | FileCheck %s --check-prefix=EMPTY --allow-empty
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s --allow-empty
// RUN: env KITRT_VERBOSE=1 %exe 2>&1 | FileCheck %s --allow-empty
//
// -----------------------------------------------------------------------------
// Log messages should be written to stderr.
//
// RUN: env KIT_VERBOSE=1 %exe 2> /dev/null \
// RUN:     | FileCheck %s --check-prefix=EMPTY --allow-empty
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 > /dev/null | FileCheck %s
//
// -----------------------------------------------------------------------------
//
// EMPTY-NOT: {{^.+$}}
//
// CHECK: test: Log message
//
// -----------------------------------------------------------------------------

#include "common/logging.h"
#include "kitrt.h"

__attribute__((constructor)) static void ctor(void) { __kitrt_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitrt_finalize(); }

int main(int argc, char *argv[]) {
  kitrt::log("test", "Log message");
  return 0;
}
