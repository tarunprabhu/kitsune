// Check that __kitser_thread_id works as expected.
//
// RUN: %exe | FileCheck %s
//
// CHECK: {{[0-9]+}}
// CHECK-NOT: {{^.+$}}

#include "serial/kitser.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitser_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitser_finalize(); }

int main(int argc, char *argv[]) {
  printf("%ld\n", __kitser_thread_id());
  return 0;
}
