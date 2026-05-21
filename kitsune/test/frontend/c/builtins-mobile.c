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

// CHECK-LABEL: allocate
// CHECK: call ptr addrspace(67) @llvm.kit.mobile.alloc(i64 {{.*}}%{{.+}})
void *__attribute__((kitsune_mobile)) allocate(unsigned long n) {
  return kitsune_mobile_alloc(n);
}

// CHECK-LABEL: deallocate
// CHECK: call void @llvm.kit.mobile.free(ptr addrspace(67) {{.*}}%{{.+}})
void deallocate(void *[[kitsune::mobile]] ptr) {
  kitsune_mobile_free(ptr);
}

// CHECK-LABEL: @cast_unsafe
// CHECK: %[[CST:.+]] = addrspacecast ptr %{{.+}} to ptr addrspace(67)
// CHECK-NEXT: ret ptr addrspace(67) %[[CST]]
void *[[kitsune::mobile]] cast_unsafe(void *ptr) {
  return __kitsune_mobile_cast_unsafe(ptr);
}
