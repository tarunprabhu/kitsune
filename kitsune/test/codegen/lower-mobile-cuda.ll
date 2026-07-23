; REQUIRES: kitsune-cuda
;
; Check basic lowering of Kitsune's mobile intrinsics with the cuda tapir
; target. These do not have any custom attributes, calling conventions, debug
; info, or metadata. Those will be checked in other tests.
;
; RUN: opt --tapir=cuda -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s

target triple = "x86_64-pc-linux-gnu"

; CHECK-LABEL: @allocate
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: %[[PTR:[0-9]+]] = call noalias ptr @__kitcuda_mem_alloc_managed(i64 %[[N]])
; CHECK-NEXT: %[[CST:[0-9]]] = addrspacecast ptr %[[PTR]] to ptr addrspace(67)
; CHECK-NEXT: ret ptr addrspace(67) %[[CST]]
define noalias ptr addrspace(67) @allocate(i64 %n) {
  %1 = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i32 2, i64 %n)
  ret ptr addrspace(67) %1
}

; CHECK-LABEL: @deallocate
; CHECK-SAME: ptr addrspace(67) %[[P:[^)]+]]
; CHECK-NEXT: %[[CST:[0-9]+]] = addrspacecast ptr addrspace(67) %[[P]] to ptr
; CHECK-NEXT: call void @__kitcuda_mem_free(ptr %[[CST]])
; CHECK-NEXT: ret void
define void @deallocate(ptr addrspace(67) %p) {
  call void @llvm.kit.mobile.free(i32 2, ptr addrspace(67) %p)
  ret void
}

; CHECK-LABEL: @fi1
; CHECK-SAME: ptr {{[^%]+}}%[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^,]+]]
; CHECK-SAME: i1 %[[INIT:[^)]+]]
; CHECK-NEXT: %[[CST:.+]] = addrspacecast {{.+}}%[[BUF]] to ptr
; CHECK-NEXT: %[[V:.+]] = zext i1 %[[INIT]] to i8
; CHECK-NEXT: call void @__kitrt_mobile_init_bool
; CHECK-SAME: (ptr {{.*}}%[[CST]], i64 %[[N]], i8 %[[V]])
define void @fi1(ptr addrspace(67) %buf, i64 %n, i1 %init) {
  call void(i32, ptr addrspace(67), i64, i1, ...) @llvm.kit.mobile.init(i32 2, ptr addrspace(67) %buf, i64 %n, i1 %init)
  ret void
}

; CHECK-LABEL: @fi8
; CHECK-SAME: ptr {{[^%]+}}%[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: %[[CST:.+]] = addrspacecast {{.+}}%[[BUF]] to ptr
; CHECK-NEXT: call void @__kitrt_mobile_init_i8
; CHECK-SAME: (ptr {{.*}}%[[CST]], i64 %[[N]], i8 8)
define void @fi8(ptr addrspace(67) %buf, i64 %n) {
  call void(i32, ptr addrspace(67), i64, i8, ...) @llvm.kit.mobile.init(i32 2, ptr addrspace(67) %buf, i64 %n, i8 8)
  ret void
}

; CHECK-LABEL: @fi16
; CHECK-SAME: ptr {{[^%]+}}%[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: %[[CST:.+]] = addrspacecast {{.+}}%[[BUF]] to ptr
; CHECK-NEXT: call void @__kitrt_mobile_init_i16
; CHECK-SAME: (ptr {{.*}}%[[CST]], i64 %[[N]], i16 16)
define void @fi16(ptr addrspace(67) %buf, i64 %n) {
  call void(i32, ptr addrspace(67), i64, i16, ...) @llvm.kit.mobile.init(i32 2, ptr addrspace(67) %buf, i64 %n, i16 16)
  ret void
}

; CHECK-LABEL: @fi32
; CHECK-SAME: ptr {{[^%]+}}%[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: %[[CST:.+]] = addrspacecast {{.+}}%[[BUF]] to ptr
; CHECK-NEXT: call void @__kitrt_mobile_init_i32
; CHECK-SAME: (ptr {{.*}}%[[CST]], i64 %[[N]], i32 32)
define void @fi32(ptr addrspace(67) %buf, i64 %n) {
  call void(i32, ptr addrspace(67), i64, i32, ...) @llvm.kit.mobile.init(i32 2, ptr addrspace(67) %buf, i64 %n, i32 32)
  ret void
}

; CHECK-LABEL: @fi64
; CHECK-SAME: ptr {{[^%]+}}%[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: %[[CST:.+]] = addrspacecast {{.+}}%[[BUF]] to ptr
; CHECK-NEXT: call void @__kitrt_mobile_init_i64
; CHECK-SAME: (ptr {{.*}}%[[CST]], i64 %[[N]], i64 64)
define void @fi64(ptr addrspace(67) %buf, i64 %n) {
  call void(i32, ptr addrspace(67), i64, i64, ...) @llvm.kit.mobile.init(i32 2, ptr addrspace(67) %buf, i64 %n, i64 64)
  ret void
}

; CHECK-LABEL: @ff32
; CHECK-SAME: ptr {{[^%]+}}%[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^,]+]]
; CHECK-SAME: float %[[INIT:[^)]+]]
; CHECK-NEXT: %[[CST:.+]] = addrspacecast {{.+}}%[[BUF]] to ptr
; CHECK-NEXT: call void @__kitrt_mobile_init_float
; CHECK-SAME: (ptr {{.*}}%[[CST]], i64 %[[N]], float %[[INIT]])
define void @ff32(ptr addrspace(67) %buf, i64 %n, float %init) {
  call void(i32, ptr addrspace(67), i64, float, ...) @llvm.kit.mobile.init(i32 2, ptr addrspace(67) %buf, i64 %n, float %init)
  ret void
}

; CHECK-LABEL: @ff64
; CHECK-SAME: ptr {{[^%]+}}%[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^,]+]]
; CHECK-SAME: double %[[INIT:[^)]+]]
; CHECK-NEXT: %[[CST:.+]] = addrspacecast {{.+}}%[[BUF]] to ptr
; CHECK-NEXT: call void @__kitrt_mobile_init_double
; CHECK-SAME: (ptr {{.*}}%[[CST]], i64 %[[N]], double %[[INIT]])
define void @ff64(ptr addrspace(67) %buf, i64 %n, double %init) {
  call void(i32, ptr addrspace(67), i64, double, ...) @llvm.kit.mobile.init(i32 2, ptr addrspace(67) %buf, i64 %n, double %init)
  ret void
}

; CHECK-LABEL: @fptr
; CHECK-SAME: ptr {{[^%]+}}%[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^,]+]]
; CHECK-SAME: ptr {{.*}}%[[INIT:[^)]+]]
; CHECK-NEXT: %[[CST:.+]] = addrspacecast {{.+}}%[[BUF]] to ptr
; CHECK-NEXT: call void @__kitrt_mobile_init_from
; CHECK-SAME: (ptr {{.*}}%[[CST]], i64 %[[N]], ptr %[[INIT]], i32 256)
define void @fptr(ptr addrspace(67) %buf, i64 %n, ptr %init) {
  call void(i32, ptr addrspace(67), i64, ptr, ...) @llvm.kit.mobile.init(i32 2, ptr addrspace(67) %buf, i64 %n, ptr %init, i32 256)
  ret void
}
