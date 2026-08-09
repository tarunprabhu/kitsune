// Check that __kitocilk_worker_id works as expected.
//
// RUN: %exe | FileCheck %s
//
// CHECK: {{[0-9]+}}
// CHECK-NOT: {{^.+$}}

#include "TestHelpers.h"
#include "opencilk/kitocilk.h"

#include <stdio.h>

CTOR(RT_OPENCILK)

int main(int argc, char *argv[]) {
  printf("%ld\n", __kitocilk_worker_id());
  return 0;
}
