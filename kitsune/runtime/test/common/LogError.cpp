// -----------------------------------------------------------------------------
// Error messages are always written out, even if verbose mode has not been
// enabled. But an error will not result in a failure.
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
// CHECK: kitrt: [test]: ERROR: Error message
//
// -----------------------------------------------------------------------------

// This must be defined before common/logging.h is included. In the actual kitrt
// source, this will have been defined by the compiler invocation in the form of
// a `-DKITRT_LOG_TAG="<...>"` command-line option.
#define KITRT_LOG_TAG "test"

#include "common/logging.h"
#include "kitrt.h"

__attribute__((constructor)) static void ctor(void) { __kitrt_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitrt_finalize(); }

int main(int argc, char *argv[]) {
  ERROR("Error message");
  return 0;
}
