// Check that the memory (de)allocation builtins are recognized and translated
// into the correct Kitsune memory (de)allocation intrinsics. This should be
// the case even if a tapir target has not been set. In C++, the builtins can be
// called directly if needed.
//
// RUN: %kitxx -ftapir=none -O1 -S -emit-llvm -o - %s | FileCheck %s
// RUN: %kitxx -O1 -S -emit-llvm -o - %s | FileCheck %s
//
// CHECK: define {{.+}} @_Z8allocateRN7kitsune
// CHECK: call ptr addrspace(67) @llvm.kit.mobile.alloc({{.+}})
//
// CHECK: define {{.+}} @_Z10deallocateRN7kitsune
// CHECK: call void @llvm.kit.mobile.free(ptr addrspace(67) %{{.+}})
//
// CHECK: define {{.+}} @allocate_c
// CHECK: call ptr addrspace(67) @llvm.kit.mobile.alloc({{.+}})
//
// CHECK: define {{.+}} @deallocate_c
// CHECK: call void @llvm.kit.mobile.free(ptr addrspace(67) %{{.+}})

#include <kitsune.h>

using namespace kitsune;

void allocate(mobile_ptr<int> &buf, size_t n) { buf.alloc(n); }

void deallocate(mobile_ptr<int> &buf) { buf.free(); }

extern "C" void *[[kitsune::mobile]] allocate_c(size_t n) {
  return kitsune_mobile_alloc(n * sizeof(int));
}

extern "C" void deallocate_c(void *[[kitsune::mobile]] ptr) {
  kitsune_mobile_free(ptr);
}
