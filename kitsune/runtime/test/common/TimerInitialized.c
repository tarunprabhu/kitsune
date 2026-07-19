// Check that __kittimer_initialized works as expected.
//
// RUN: %exe | FileCheck %s
//
// CHECK: Before initialize: 0
// CHECK: After initialize: {{[1-9][0-9]*}}
// CHECK: Before finalize: {{[1-9][0-9]*}}
// CHECK: After finalize: 0

#include "common/timer.h"
#include "kitrt.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) {
  // We call kitrt_initialize because that will call __kittimer_initialize. We
  // also require the global context to be initialized correctly before
  // __kittimer_initialize is called.
  printf("Before initialize: %d\n", __kittimer_initialized());
  __kitrt_initialize();
  printf("After initialize: %d\n", __kittimer_initialized());
}

__attribute__((destructor)) static void dtor(void) {
  // __kitrt_finalize will call __kittimer_finalize. We need to finalize the
  // runtime anyway, so this accomplishes both requirements.
  printf("Before finalize: %d\n", __kittimer_initialized());
  __kitrt_finalize();
  printf("After finalize: %d\n", __kittimer_initialized());
}

int main(int argc, char *argv[]) { return 0; }
