// RUN: %kitcc -ftapir=hip -O2 -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DECLARES

#include <kitsune.h>

void* allocate(size_t n) {
  return kitsune_mobile_alloc(n * sizeof(int));
}

// CHECK-LABEL: allocate
// CHECK: call {{.+}} @__kithip_mem_alloc_managed({{.+}})

void deallocate(void* ptr) {
  kitsune_mobile_free(ptr);
}

// CHECK-LABEL: deallocate
// CHECK: call {{.+}} @__kithip_mem_free({{.+}})

// DECLARES-DAG: declare noalias ptr @__kithip_mem_alloc_managed
