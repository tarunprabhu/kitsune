// Check that initializing the runtime multiple times has no effect. The
// runtime should remain initialized.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initialized = 0
// CHECK: Initializing Kitsune runtime (openmp)
// CHECK: Initializing Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (openmp)
// CHECK: Initialized = {{[1-9][0-9]*}}
// CHECK: Runtime already initialized
// CHECK: Initialized = {{[1-9][0-9]*}}

#include <openmp/kitomp.h>

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) {
  fprintf(stderr, "Initialized = %d\n", __kitomp_initialized());
  __kitomp_initialize();
  fprintf(stderr, "Initialized = %d\n", __kitomp_initialized());
  __kitomp_initialize();
  fprintf(stderr, "Initialized = %d\n", __kitomp_initialized());
}

__attribute__((destructor)) static void dtor(void) { __kitomp_finalize(); }

int main(int argc, char *argv[]) { return 0; }
