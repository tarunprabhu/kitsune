// REQUIRES: amd-gpu
//
// Check that __kithip_is_initialized works as expected.
//
// RUN: %exe | FileCheck %s
//
// CHECK: Before initialize: 0
// CHECK: After initialize: {{[1-9][0-9]*}}
// CHECK: Before finalize: {{[1-9][0-9]*}}
// CHECK: After finalize: 0

#include <stdbool.h>
#include <stdio.h>

// hip/kithip.h is not safe to be included in C source files. That header
// should be modified, but that might be a non-trivial change, so for now, this
// is easier.
bool __kithip_is_initialized(void);
void __kithip_initialize(void);
void __kithip_finalize(void);

__attribute__((constructor)) static void ctor(void) {
  printf("Before initialize: %d\n", __kithip_is_initialized());
  __kithip_initialize();
  printf("After initialize: %d\n", __kithip_is_initialized());
}

__attribute__((destructor)) static void dtor(void) {
  printf("Before finalize: %d\n", __kithip_is_initialized());
  __kithip_finalize();
  printf("After finalize: %d\n", __kithip_is_initialized());
}

int main(int argc, char *argv[]) { return 0; }
