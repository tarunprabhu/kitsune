// If no events are measured, nothing should be printed at runtime.
//
// RUN: %exe 2>&1 | FileCheck %s --allow-empty
//
// CHECK-NOT: {{^.+$}}

#include "common/kitpapi.h"
#include "kitrt.h"

#include <stddef.h>

__attribute__((constructor)) static void ctor(void) {
  __kitrt_initialize();
  __kitpapi_initialize(NULL);
}

__attribute__((destructor)) static void dtor(void) {
  __kitpapi_finalize();
  __kitrt_finalize();
}

int main(int argc, char *argv[]) { return 0; }
