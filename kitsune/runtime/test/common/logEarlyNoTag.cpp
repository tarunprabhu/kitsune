// -----------------------------------------------------------------------------
// logEarly works with both KIT_VERBOSE and KITRT_VERBOSE.
//
// RUN: %exe 2>&1 | FileCheck %s --check-prefix=EMPTY --allow-empty
// RUN: env KIT_VERBOSE=1 %exe 2>&1 \
// RUN:     | FileCheck %s --match-full-lines
// RUN: env KIT_VERBOSE=1 %exe 2>&1 \
// RUN:     | FileCheck %s --match-full-lines
//
// -----------------------------------------------------------------------------
// If either KIT_VERBOSE, or KITRT_VERBOSE is set, then early log messages will
// be written.
//
// RUN: env KIT_VERBOSE=0 KITRT_VERBOSE=1 %exe 2>&1 \
// RUN:     | FileCheck %s --match-full-lines
// RUN: env KIT_VERBOSE=1 KITRT_VERBOSE=0 %exe 2>&1 \
// RUN:     | FileCheck %s --match-full-lines
//
// RUN: env KIT_VERBOSE=0 KITRT_VERBOSE=0 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=EMPTY --allow-empty
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
// CHECK: kitrt: Early message
//
// -----------------------------------------------------------------------------

#include "common/logging.h"
#include "kitrt.h"

__attribute__((constructor)) static void ctor(void) { __kitrt_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitrt_finalize(); }

int main(int argc, char *argv[]) {
  LOGEARLY("Early message");
  return 0;
}
