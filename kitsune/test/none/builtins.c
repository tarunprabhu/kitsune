// With the none tapir target, nothing is lowered, including the Kitsune
// builtins/intrinsics. It is not clear if this is what we want.

// RUN: %kitcc -ftapir=none -std=c23 -O2 -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefixes=DECLARES,CHECK

#include <kitsune.h>

void *[[kitsune::mobile]] allocate(size_t n) {
  return kitsune_mobile_alloc(n * sizeof(int));
}

// CHECK-LABEL: allocate
// CHECK: call {{.+}} @llvm.kitsune.mobile.alloc({{.+}})

// DECLARES-DAG: declare noalias ptr addrspace(67) @llvm.kitsune.mobile.alloc

void deallocate(void *[[kitsune::mobile]] buf) { kitsune_mobile_free(buf); }

// CHECK-LABEL: deallocate
// CHECK: call {{.+}} @llvm.kitsune.mobile.free({{.+}})
