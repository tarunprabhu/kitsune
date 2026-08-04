// REQUIRES: amd-gpu
//
// Check that finalizing the runtime multiple times has no effect. The runtime
// should remain finalized.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initialized = {{[1-9][0-9]*}}
// CHECK: Finalizing Kitsune runtime (hip)
// CHECK: Finalized Kitsune runtime (hip)
// CHECK: Finalizing Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (common)
// CHECK: Initialized = 0
// CHECK: Cannot finalize runtime. Not initialized
// CHECK: Initialized = 0

#include <stdbool.h>
#include <stdio.h>

// hip/kithip.h pulls in the hip headers. Since we don't need any of that, just
// declare what we need.
extern "C" bool __kithip_is_initialized(void);
extern "C" void __kithip_initialize(void);
extern "C" void __kithip_finalize(void);

__attribute__((constructor)) static void ctor(void) { __kithip_initialize(); }

__attribute__((destructor)) static void dtor(void) {
  fprintf(stderr, "Initialized = %d\n", __kithip_is_initialized());
  __kithip_finalize();
  fprintf(stderr, "Initialized = %d\n", __kithip_is_initialized());
  __kithip_finalize();
  fprintf(stderr, "Initialized = %d\n", __kithip_is_initialized());
}

int main(int argc, char *argv[]) { return 0; }
