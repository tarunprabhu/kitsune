// Check that __kitser_initialized works as expected.
//
// RUN: %exe | FileCheck %s
//
// CHECK: Before initialize: 0
// CHECK: After initialize: {{[1-9][0-9]*}}
// CHECK: Before finalize: {{[1-9][0-9]*}}
// CHECK: After finalize: 0

#include "serial/kitser.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) {
  printf("Before initialize: %d\n", __kitser_initialized());
  __kitser_initialize();
  printf("After initialize: %d\n", __kitser_initialized());
}

__attribute__((destructor)) static void dtor(void) {
  printf("Before finalize: %d\n", __kitser_initialized());
  __kitser_finalize();
  printf("After finalize: %d\n", __kitser_initialized());
}

int main(int argc, char *argv[]) { return 0; }
