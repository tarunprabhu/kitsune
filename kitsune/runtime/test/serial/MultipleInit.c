// Check that initializing the runtime multiple times has no effect. The
// runtime should remain initialized.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initialized = 0
// CHECK: Initializing Kitsune runtime (serial)
// CHECK: Initializing Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (serial)
// CHECK: Initialized = {{[1-9][0-9]*}}
// CHECK: Runtime already initialized
// CHECK: Initialized = {{[1-9][0-9]*}}

#include <serial/kitser.h>

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) {
  fprintf(stderr, "Initialized = %d\n", __kitser_initialized());
  __kitser_initialize();
  fprintf(stderr, "Initialized = %d\n", __kitser_initialized());
  __kitser_initialize();
  fprintf(stderr, "Initialized = %d\n", __kitser_initialized());
}

__attribute__((destructor)) static void dtor(void) { __kitser_finalize(); }

int main(int argc, char *argv[]) { return 0; }
