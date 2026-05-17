// Check that the memory (de)allocation builtins are recognized and translated
// to the correct Kitsune memory (de)allocation intrinsics. This should be done
// even if a tapir target has not been set.
//
// FIXME: The intrinsics should *ONLY* be recognized when the kitcc driver is
// used. It should not work with the 'clang' driver.
//
// RUN: %clang -O0 -S -emit-llvm -o - %s %sysroot | FileCheck %s
// RUN: %kitcc -O0 -S -emit-llvm -o - %s %sysroot | FileCheck %s
// RUN: %kitcc --tapir=nolo -O0 -S -emit-llvm -o - %s %sysroot | FileCheck %s

#include <stddef.h>

// CHECK-LABEL: allocate_c
// CHECK: call ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %{{.+}})
void *__attribute__((kitsune_mobile)) allocate_c(size_t n) {
  return kitsune_mobile_alloc(n);
}

// CHECK-LABEL: deallocate_c
// CHECK: call void @llvm.kit.mobile.free(ptr addrspace(67) {{.*}}%{{.+}})
void deallocate_c(void *[[kitsune::mobile]] ptr) {
  kitsune_mobile_free(ptr);
}
