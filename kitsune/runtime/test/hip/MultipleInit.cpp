// REQUIRES: amd-gpu
//
// Check that initializing the runtime multiple times has no effect. The
// runtime should remain initialized.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initialized = 0
// CHECK: Initializing Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (common)
// CHECK: Initializing Kitsune runtime (hip)
// CHECK: Initialized Kitsune runtime (hip)
// CHECK: Initialized = {{[1-9][0-9]*}}
// CHECK: Runtime already initialized
// CHECK: Initialized = {{[1-9][0-9]*}}

#include <stdbool.h>
#include <stdio.h>

// hip/kithip.h pulls in the hip headers. Since we don't need any of that, just
// declare what we need.
extern "C" bool __kithip_is_initialized(void);
extern "C" void __kithip_initialize(void);
extern "C" void __kithip_finalize(void);

__attribute__((constructor)) static void ctor(void) {
  fprintf(stderr, "Initialized = %d\n", __kithip_is_initialized());
  __kithip_initialize();
  fprintf(stderr, "Initialized = %d\n", __kithip_is_initialized());
  __kithip_initialize();
  fprintf(stderr, "Initialized = %d\n", __kithip_is_initialized());
}

__attribute__((destructor)) static void dtor(void) { __kithip_finalize(); }

int main(int argc, char *argv[]) { return 0; }
