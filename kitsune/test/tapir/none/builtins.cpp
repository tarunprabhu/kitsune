// With the none tapir target, nothing is lowered, including the Kitsune
// builtins/intrinsics. It is not clear if this is what we want.

// RUN: %kitxx -ftapir=none -O2 -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefixes=CHECK,DECLARES

#include <kitsune.h>

using namespace kitsune;

void allocate(mobile_ptr<int>& buf, size_t n) {
  buf.alloc(n);
}

// CHECK-LABEL: _Z8allocate
// CHECK: call {{.+}} @llvm.kitsune.mobile.alloc({{.+}})

void deallocate(mobile_ptr<int>& buf) {
  buf.free();
}

// CHECK-LABEL: _Z10deallocate
// CHECK: call {{.+}} @llvm.kitsune.mobile.free({{.+}})

// DECLARES-DAG: declare noalias ptr addrspace(67) @llvm.kitsune.mobile.alloc
