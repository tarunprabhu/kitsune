// Check that the memory (de)allocation builtins are recognized and translated
// into the correct Kitsune memory (de)allocation intrinsics. This should be
// the case even if a tapir target has not been set.
//
// RUN: %clang -O0 -S -emit-llvm -o - %s | FileCheck  %s
// RUN: %kitcc -ftapir=none -O0 -S -emit-llvm -o - %s | FileCheck %s
// RUN: %kitcc -O0 -S -emit-llvm -o - %s | FileCheck %s
//
// CHECK-LABEL: allocate_c
// CHECK: call ptr addrspace(67) @llvm.kit.mobile.alloc({{.+}})
//
// CHECK-LABEL: deallocate_c
// CHECK: call void @llvm.kit.mobile.free(ptr addrspace(67) %{{.+}})

#include "kitsune.h"

void *__attribute__((kitsune_mobile)) allocate_c(size_t n) {
  return kitsune_mobile_alloc(n * sizeof(int));
}

void deallocate_c(void *[[kitsune::mobile]] ptr) {
  kitsune_mobile_free(ptr);
}
