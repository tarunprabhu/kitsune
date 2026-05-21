// Check that the memory (de)allocation builtins are recognized and translated
// into the correct Kitsune memory (de)allocation intrinsics. This should be
// the case even if a tapir target has not been set. In C++, the builtins can be
// called directly if needed.
//
// FIXME: The builtins should *ONLY* be recognized when the kit++ driver is
// used. It should not work with the 'clang' driver.
//
// RUN: %clangxx -O0 -S -emit-llvm -o - %s %sysroot | FileCheck %s
// RUN: %kitxx -O0 -S -emit-llvm -o - %s %sysroot | FileCheck %s
// RUN: %kitxx --tapir=nolo -O0 -S -emit-llvm -o - %s %sysroot | FileCheck %s

// CHECK-LABEL: @allocate
// CHECK: call ptr addrspace(67) @llvm.kit.mobile.alloc({{.+}})
extern "C" void *[[kitsune::mobile]] allocate(unsigned long n) {
  return kitsune_mobile_alloc(n * sizeof(int));
}

// CHECK-LABEL: @deallocate
// CHECK: call void @llvm.kit.mobile.free(ptr addrspace(67) %{{.+}})
extern "C" void deallocate(void *[[kitsune::mobile]] ptr) {
  kitsune_mobile_free(ptr);
}

// CHECK-LABEL: @cast_unsafe
// CHECK: %[[CST:.+]] = addrspacecast ptr %{{.+}} to ptr addrspace(67)
// CHECK-NEXT: ret ptr addrspace(67) %[[CST]]
extern "C" void *[[kitsune::mobile]] cast_unsafe(void *ptr) {
  return __kitsune_mobile_cast_unsafe(ptr);
}
