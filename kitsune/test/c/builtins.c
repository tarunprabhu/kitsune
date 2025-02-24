// The serial target is always built, so this is safe in all builds.
// RUN: %kitcc -ftapir=serial -O2 -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefixes=TAPIR,DECLARES

// If the builtins are used, they should be lowered correctly even if no tapir
// target is enabled. In this case, it should directly call malloc/free since
// the kitrt runtime will not be linked.
// RUN: %kitcc -O0 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=NOTAPIR
// RUN: %kitcc -O2 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=NOTAPIR

#include <kitsune.h>

void *__attribute__((kitsune_mobile)) allocate_c(size_t n) {
  return kitsune_mobile_alloc(n * sizeof(int));
}

// TAPIR-LABEL: allocate_c
// TAPIR: call {{.+}} @malloc({{.+}})
// NOTAPIR: call {{.+}} @malloc({{.+}})

void deallocate_c(void *[[kitsune::mobile]] ptr) { kitsune_mobile_free(ptr); }

// TAPIR-LABEL: deallocate_c
// TAPIR: call {{.+}} @free({{.+}})
// NOTAPIR: call {{.+}} @free({{.+}})

// DECLARES-DAG: declare noalias ptr addrspace(67) @malloc
