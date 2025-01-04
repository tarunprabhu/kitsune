// RUN: %kitcc -std=c23 -ftapir=serial -O2 -S -emit-llvm -o - %s | FileCheck %s
// --check-prefixes=CHECK,DECLARES

#include <kitsune.h>

void *[[kitsune::mobile]] allocate(size_t n) {
  return kitsune_mobile_alloc(n * sizeof(int));
}

// CHECK-LABEL: allocate
// CHECK: call {{.+}} @__kitrt_default_mem_alloc({{.+}})

void deallocate(void *[[kitsune::mobile]] ptr) { kitsune_mobile_free(ptr); }

// CHECK-LABEL: deallocate
// CHECK: call {{.+}} @__kitrt_default_mem_free({{.+}})

// DECLARES-DAG: declare noalias ptr addrspace(67) @__kitrt_default_mem_alloc
