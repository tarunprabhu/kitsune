// If nothing is timed, nothing should be printed.
//
// RUN: %exe 2>&1 | FileCheck %s --allow-empty
//
// CHECK-NOT: {{^.+$}}

#include "kitrt.h"

__attribute__((constructor)) static void ctor(void) { __kitrt_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitrt_finalize(); }

int main(int argc, char *argv[]) { return 0; }
