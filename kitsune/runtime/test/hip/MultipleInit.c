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

// hip/kithip.h is not safe to be included in C source files. That header should
// be modified, but that might be a non-trivial change, so for now, this is
// easier.
bool __kithip_is_initialized(void);
void __kithip_initialize(void);
void __kithip_finalize(void);

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) {
  fprintf(stderr, "Initialized = %d\n", __kithip_is_initialized());
  __kithip_initialize();
  fprintf(stderr, "Initialized = %d\n", __kithip_is_initialized());
  __kithip_initialize();
  fprintf(stderr, "Initialized = %d\n", __kithip_is_initialized());
}

__attribute__((destructor)) static void dtor(void) { __kithip_finalize(); }

int main(int argc, char *argv[]) { return 0; }
