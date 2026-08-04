// REQUIRES: kitsune-papi
//
// If no hardware counters are collected using PAPI, snothing should be written
// to the specified counters file. The counters file should not be created.
//
// RUN: rm -rf %t.json
// RUN: env KIT_PAPI_FILE=%t.json %exe | FileCheck %s --allow-empty
// RUN: not cat %t.json | FileCheck %s --check-prefix=NOEXIST --allow-empty
//
// NOEXIST-NOT: {{^.+$}}
//
// -----------------------------------------------------------------------------
// If the file already exists, it must remain unchanged.
//
// RUN: echo "Contents" > %t.existing.json
// RUN: env KIT_PAPI_FILE=%t.json %exe | FileCheck %s --allow-empty
// RUN: cat %t.existing.json | FileCheck %s --check-prefix=UNCHANGED
//
// UNCHANGED: Contents
// UNCHANGED-NOT: {{^.+$}}
//
// CHECK-NOT: {{^.+$}}
//
// -----------------------------------------------------------------------------

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
