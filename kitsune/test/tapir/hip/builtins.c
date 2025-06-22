// RUN: %kitcc -ftapir=hip -std=c23 -O2 -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefixes=CHECK,DECLARES

#include <kitsune.h>

void *[[kitsune::mobile]] allocate(size_t n) {
  return kitsune_mobile_alloc(n * sizeof(int));
}

// CHECK-LABEL: allocate
// CHECK: call {{.+}} @__kithip_mem_alloc_managed({{.+}})

void deallocate(void *[[kitsune::mobile]] ptr) { kitsune_mobile_free(ptr); }

// CHECK-LABEL: deallocate
// CHECK: call {{.+}} @__kithip_mem_free({{.+}})

// DECLARES-DAG: declare noalias ptr addrspace(67) @__kithip_mem_alloc_managed
