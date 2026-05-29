// Check that the memory (de)allocation builtins are recognized and translated
// to the correct Kitsune memory (de)allocation intrinsics. This should be done
// even if a tapir target has not been set.
//
// FIXME: The builtins should *ONLY* be recognized when the kit++ driver is
// used. It should not work with the 'clang' driver.
//
// RUN: %clangxx -O0 -S -emit-llvm -o - %s %sysroot \
// RUN:     -Xclang -disable-llvm-passes \
// RUN:     | FileCheck %s --check-prefixes=ALL,UNSPECIFIED
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

// ALL-LABEL: allocate
// ALL-SAME: i64 {{.*}}%[[ARGN:[^)]+]]
// ALL-NEXT: [[ENTRY:.+]]:
// ALL-NEXT: %[[SLOT:.+]] = alloca i64
// ALL-NEXT: store i64 %[[ARGN]], ptr %[[SLOT]]
// ALL-NEXT: %[[N:.+]] = load i64, ptr %[[SLOT]]
// UNSPECIFIED: call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1, i64 %[[N]])
// NOLO: call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 0, i64 %[[N]])
// SERIAL: call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1, i64 %[[N]])
// PTHREADS: call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1024, i64 %[[N]])
extern "C" void *[[kitsune::mobile]] allocate(unsigned long n) {
  return kitsune_mobile_alloc(n);
}

// ALL-LABEL: deallocate
// ALL-SAME: ptr addrspace(67) {{.*}}%[[ARGPTR:[^)]+]]
// ALL-NEXT: [[ENTRY:.+]]:
// ALL-NEXT: %[[SLOT:.+]] = alloca ptr addrspace(67)
// ALL-NEXT: store ptr addrspace(67) %[[ARGPTR]], ptr %[[SLOT]]
// ALL-NEXT: %[[PTR:.+]] = load ptr addrspace(67), ptr %[[SLOT]]
// UNSPECIFIED: call void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) %[[PTR]])
// NOLO: call void @llvm.kit.mobile.free(i32 0, ptr addrspace(67) %[[PTR]])
// SERIAL: call void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) %[[PTR]])
// PTHREADS: call void @llvm.kit.mobile.free(i32 1024, ptr addrspace(67) %[[PTR]])
extern "C" void deallocate(void *[[kitsune::mobile]] ptr) {
  kitsune_mobile_free(ptr);
}

// ALL-LABEL: @cast_unsafe
// ALL: %[[CST:.+]] = addrspacecast ptr %{{.+}} to ptr addrspace(67)
// ALL-NEXT: ret ptr addrspace(67) %[[CST]]
extern "C" void *[[kitsune::mobile]] cast_unsafe(void *ptr) {
  return __kitsune_mobile_cast_unsafe(ptr);
}
