// Check that the memory (de)allocation builtins are recognized and translated
// into the correct Kitsune memory (de)allocation intrinsics. This should be
// the case even if a tapir target has not been set. In C++, the builtins can be
// called directly if needed.
//
// RUN: %clangxx -O0 -S -emit-llvm -o - %s %sysroot | FileCheck %s
// RUN: %kitxx -O0 -S -emit-llvm -o - %s %sysroot | FileCheck %s
// RUN: %kitxx --tapir=nolo -O0 -S -emit-llvm -o - %s %sysroot | FileCheck %s

#include <kitsune.h>

// CHECK-LABEL: @_Z8allocateRN7kitsune
// CHECK: call ptr addrspace(67) @llvm.kit.mobile.alloc({{.+}})
void allocate(kitsune::mobile_ptr<int> &buf, size_t n) { buf.alloc(n); }

// CHECK-LABEL: @_Z10deallocateRN7kitsune
// CHECK: call void @llvm.kit.mobile.free(ptr addrspace(67) %{{.+}})
void deallocate(kitsune::mobile_ptr<int> &buf) { buf.free(); }

// CHECK-LABEL: @allocate_c
// CHECK: call ptr addrspace(67) @llvm.kit.mobile.alloc({{.+}})
extern "C" void *[[kitsune::mobile]] allocate_c(size_t n) {
  return kitsune_mobile_alloc(n * sizeof(int));
}

// CHECK-LABEL: @deallocate_c
// CHECK: call void @llvm.kit.mobile.free(ptr addrspace(67) %{{.+}})
extern "C" void deallocate_c(void *[[kitsune::mobile]] ptr) {
  kitsune_mobile_free(ptr);
}
