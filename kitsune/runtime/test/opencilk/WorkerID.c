// Check that __kitocilk_worker_id works as expected.
//
// RUN: %exe | FileCheck %s
//
// CHECK: {{[0-9]+}}
// CHECK-NOT: {{^.+$}}

#include "opencilk/kitocilk.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitocilk_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitocilk_finalize(); }

int main(int argc, char *argv[]) {
  printf("%ld\n", __kitocilk_worker_id());
  return 0;
}
