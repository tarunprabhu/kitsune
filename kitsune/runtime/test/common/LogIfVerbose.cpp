// -----------------------------------------------------------------------------
// logIfVerbose works with both KIT_VERBOSE and KITRT_VERBOSE.
//
// RUN: %exe 2>&1 | FileCheck %s --check-prefix=EMPTY --allow-empty
// RUN: env KIT_VERBOSE=1 %exe 2>&1 \
// RUN:     | FileCheck %s --match-full-lines
// RUN: env KIT_VERBOSE=1 %exe 2>&1 \
// RUN:     | FileCheck %s --match-full-lines
//
// -----------------------------------------------------------------------------
// KIT_VERBOSE takes priority over KITRT_VERBOSE.
//
// RUN: env KIT_VERBOSE=0 KITRT_VERBOSE=0 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=EMPTY --allow-empty
// RUN: env KIT_VERBOSE=0 KITRT_VERBOSE=1 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=EMPTY --allow-empty
// RUN: env KIT_VERBOSE=1 KITRT_VERBOSE=0 %exe 2>&1 \
// RUN:     | FileCheck %s --match-full-lines
// RUN: env KIT_VERBOSE=1 KITRT_VERBOSE=1 %exe 2>&1 \
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
// CHECK: kitrt: [test]: Early message
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
  LOG_IF_VERBOSE("Early message");
  return 0;
}
