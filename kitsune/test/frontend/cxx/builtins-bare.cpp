// Check that the temporary memory (de)allocation builtins are recognized and
// translated to the correct Kitsune memory (de)allocation intrinsics. This
// should be done even if a tapir target has not been set.
//
// RUN: %kitxx -O0 -S -emit-llvm -o - %s %sysroot \
// RUN:     | FileCheck %s
//
// RUN: %kitxx --tapir=nolo -O0 -S -emit-llvm -o - %s %sysroot \
// RUN:     | FileCheck %s

#include "kitsune.h"

// CHECK-LABEL: allocate_bare_c
// CHECK: %[[PTR:.+]] = call ptr addrspace(67) @llvm.kit.mobile.alloc({{.+}})
// CHECK-NEXT: store ptr addrspace(67) %[[PTR]], ptr %[[SLOT:[^,]+]]
// CHECK-NEXT: %[[RET:.+]] = load ptr, ptr %[[SLOT]]
// CHECK-NEXT: ret ptr %[[RET]]
void *allocate_bare_cxx(size_t n) { return kitsune_mobile_alloc__(n); }

// CHECK-LABEL: deallocate_bare_c
// CHECK: %[[MOBILE:.+]] = addrspacecast ptr %{{.+}} to ptr addrspace(67)
// CHECK-NEXT: call void @llvm.kit.mobile.free(ptr addrspace(67) %[[MOBILE]])
void deallocate_bare_cxx(void *ptr) { kitsune_mobile_free__(ptr); }
