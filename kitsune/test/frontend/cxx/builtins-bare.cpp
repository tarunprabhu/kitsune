// Check that the IR generated for the memory (de)allocation functions that
// return a regular pointer (as opposed to a mobile pointer) are as expected.
// These are wrappers around Kitsune's builtins, but are defined in kitsune.h.
// In other words, they are not builtins.
//
// RUN: %kitxx -O1 -S -emit-llvm -o - %s %sysroot \
// RUN:     -Xclang -disable-llvm-passes \
// RUN:     | FileCheck %s --check-prefixes=ALL,UNSPECIFIED
//
// RUN: %kitxx --tapir=nolo -O1 -S -emit-llvm -o - %s %sysroot \
// RUN:     -Xclang -disable-llvm-passes \
// RUN:     | FileCheck %s --check-prefixes=ALL,NOLO
//
// RUN: %kitxx --tapir=serial -O1 -S -emit-llvm -o - %s %sysroot \
// RUN:     -Xclang -disable-llvm-passes \
// RUN:     | FileCheck %s --check-prefixes=ALL,SERIAL
//
// RUN: %kitxx --tapir=pthreads -O1 -S -emit-llvm -o - %s %sysroot \
// RUN:     -Xclang -disable-llvm-passes \
// RUN:     | FileCheck %s --check-prefixes=ALL,PTHREADS

#include "kitsune.h"

// ALL-LABEL: define {{.+}} @kitsune_mobile_alloc__(
// ALL-SAME: i64 {{.*}}%[[ARGN:[^)]+]]
// ALL-NEXT: [[ENTRY:.+]]:
// ALL-NEXT: %[[RETSLOT:.+]] = alloca ptr
// ALL-NEXT: %[[ARGSLOT:.+]] = alloca i64
// ALL-NEXT: store i64 %[[ARGN]], ptr %[[ARGSLOT]]
// ALL-NEXT: %[[N:.+]] = load i64, ptr %[[ARGSLOT]]
// UNSPECIFIED-NEXT: %[[PTR:.+]] = call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1, i64 %[[N]])
// NOLO-NEXT: %[[PTR:.+]] = call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 0, {{.+}})
// SERIAL-NEXT: %[[PTR:.+]] = call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1, i64 %[[N]])
// PTHREADS-NEXT: %[[PTR:.+]] = call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1024, i64 %[[N]])
// ALL-NEXT: store ptr addrspace(67) %[[PTR]], ptr %[[RETSLOT]]
// ALL-NEXT: %[[RET:.+]] = load ptr, ptr %[[RETSLOT]]
// ALL-NEXT: ret ptr %[[RET]]

// ALL-LABEL: define {{.+}} @kitsune_mobile_free__(
// ALL-SAME: ptr {{.*}}%[[ARGPTR:[^)]+]]
// ALL-NEXT: [[ENTRY:.+]]:
// ALL-NEXT: %[[SLOT:.+]] = alloca ptr
// ALL-NEXT: store ptr %[[ARGPTR]], ptr %[[SLOT]]
// ALL-NEXT: %[[PTR:.+]] = load ptr, ptr %[[SLOT]]
// ALL-NEXT: %[[MOBILE:.+]] = addrspacecast ptr %[[PTR]] to ptr addrspace(67)
// UNSPECIFIED-NEXT: call void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) %[[MOBILE]])
// NOLO-NEXT: call void @llvm.kit.mobile.free(i32 0, ptr addrspace(67) %[[MOBILE]])
// SERIAL-NEXT: call void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) %[[MOBILE]])
// PTHREADS-NEXT: call void @llvm.kit.mobile.free(i32 1024, ptr addrspace(67) %[[MOBILE]])

extern "C" void *allocate(size_t n) { return kitsune_mobile_alloc__(n); }

extern "C" void deallocate(void *ptr) { kitsune_mobile_free__(ptr); }
