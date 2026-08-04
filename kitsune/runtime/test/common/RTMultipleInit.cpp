// Check that initializing the common runtime multiple times has no effect.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initializing Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (common)
// CHECK: Runtime already initialized
// CHECK-NOT: Initializing Kitsune runtime

#include <openmp/kitomp.h>

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) {
  __kitomp_initialize();
  __kitomp_initialize();
}

__attribute__((destructor)) static void dtor(void) { __kitomp_finalize(); }

int main(int argc, char *argv[]) { return 0; }
