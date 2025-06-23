; RUN: opt -passes "strip-kitsune-addrspace" -S %s | FileCheck %s

source_filename = "-"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; CHECK: %S1{{.*}} = type { i32, ptr }
%S1 = type { i32, ptr addrspace(67) }

; CHECK: @g1 = external global ptr
@g1 = external global ptr addrspace(67)

; CHECK: @g2 = global ptr null
@g2 = global ptr addrspace(67) null

; CHECK: @g3 = global [2 x ptr] [ptr null, ptr inttoptr (i64 1234 to ptr)]
@g3 = global [2 x ptr addrspace(67)] [ptr addrspace(67) null, ptr addrspace(67) inttoptr (i64 1234 to ptr addrspace(67))]

; CHECK @g4 = external global %S1{{.*}}
@g4 = external global %S1

; CHECK: declare ptr @f0(ptr, ptr)
declare ptr addrspace(67) @f0(ptr, ptr addrspace(67))

; CHECK: define void @f1()
define void @f1() {
  ret void
}

; CHECK: define void @f2(ptr %0)
define void @f2(ptr %0) {
  ret void
}

; CHECK: define void @f3(ptr %0)
define void @f3(ptr addrspace(67) %0) {
  ret void
}

; CHECK: define ptr @f4(ptr %0)
; CHECK-NEXT: ret ptr %0
define ptr addrspace(67) @f4(ptr addrspace(67) %0) {
  ret ptr addrspace(67) %0
}

; CHECK: define ptr @f5()
; CHECK-NEXT: ret ptr null
define ptr addrspace(67) @f5() {
  ret ptr addrspace(67) null
}

; CHECK: define i32 @f6(i32 %0)
; CHECK-NEXT: %2 = alloca i32
; CHECK-NEXT: store i32 %0, ptr %2
; CHECK-NEXT: %3 = load i32, ptr %2
; CHECK-NEXT: ret i32 %3
define i32 @f6(i32 %0) {
  %2 = alloca i32, addrspace(67)
  store i32 %0, ptr addrspace(67) %2
  %3 = load i32, ptr addrspace(67) %2
  ret i32 %3
}

; CHECK: define float @f7()
; CHECK-NEXT: %1 = alloca float
; CHECK-NEXT: %2 = call ptr @f4(ptr %1)
; CHECK-NEXT: %3 = load float, ptr %2
; CHECK-NEXT: ret float %3
define float @f7() {
  %1 = alloca float, addrspace(67)
  %2 = call ptr addrspace(67) @f4(ptr addrspace(67) %1)
  %3 = load float, ptr addrspace(67) %2
  ret float %3
}

; CHECK: define ptr @f8(ptr %0)
; CHECK-NEXT: ret ptr %0
define ptr addrspace(67) @f8(ptr %0) {
  %2 = addrspacecast ptr %0 to ptr addrspace(67)
  ret ptr addrspace(67) %2
}

; CHECK: define ptr @f9(ptr %0)
; CHECK-NEXT: ret ptr %0
define ptr @f9(ptr %0) {
  %2 = addrspacecast ptr %0 to ptr addrspace(67)
  %3 = addrspacecast ptr addrspace(67) %2 to ptr
  ret ptr %3
}

; CHECK: define ptr @f10(ptr %0)
; CHECK-NEXT: %2 = addrspacecast ptr %0 to ptr addrspace(3)
; CHECK-NEXT: %3 = addrspacecast ptr addrspace(3) %2 to ptr
; CHECK-NEXT: ret ptr %3
define ptr @f10(ptr %0) {
  %2 = addrspacecast ptr %0 to ptr addrspace(3)
  %3 = addrspacecast ptr addrspace(3) %2 to ptr addrspace(67)
  %4 = addrspacecast ptr addrspace(67) %3 to ptr
  ret ptr %4
}

; CHECK-LABEL: @fix_intrinsics
; CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32
; CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64
; CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32
; CHECK-NEXT: call void @llvm.memcpy.inline.p0.p0.i32
; CHECK-NEXT: call void @llvm.memcpy.inline.p0.p0.i64
; CHECK-NEXT: call void @llvm.memcpy.inline.p0.p0.i32
; CHECK-NEXT: call void @llvm.memmove.p0.p0.i32
; CHECK-NEXT: call void @llvm.memmove.p0.p0.i64
; CHECK-NEXT: call void @llvm.memmove.p0.p0.i32
; CHECK-NEXT: call void @llvm.memmove.p0.p0.i32
; CHECK-NEXT: call void @llvm.memmove.p0.p0.i64
; CHECK-NEXT: call void @llvm.memmove.p0.p0.i32
; CHECK-NEXT: call void @llvm.memset.p0.i32
; CHECK-NEXT: call void @llvm.memset.p0.i64
; CHECK-NEXT: call void @llvm.memset.inline.p0.i32
; CHECK-NEXT: call void @llvm.memset.inline.p0.i64
define void @fix_intrinsics(ptr addrspace(67) %0, ptr addrspace(67) %1, ptr %2, i32 %n4, i64 %n8) {
  call void @llvm.memcpy.p0.p67.i32(ptr %2, ptr addrspace(67) %0, i32 %n4, i1 true)
  call void @llvm.memcpy.p67.p67.i64(ptr addrspace(67) %0, ptr addrspace(67) %1, i64 %n8, i1 true)
  call void @llvm.memcpy.p67.p0.i32(ptr addrspace(67) %0, ptr %2, i32 %n4, i1 true)
  call void @llvm.memcpy.inline.p0.p67.i32(ptr %2, ptr addrspace(67) %0, i32 %n4, i1 true)
  call void @llvm.memcpy.inline.p67.p67.i64(ptr addrspace(67) %0, ptr addrspace(67) %1, i64 %n8, i1 true)
  call void @llvm.memcpy.inline.p67.p0.i32(ptr addrspace(67) %0, ptr %2, i32 %n4, i1 true)

  call void @llvm.memmove.p0.p67.i32(ptr %2, ptr addrspace(67) %0, i32 %n4, i1 true)
  call void @llvm.memmove.p67.p67.i64(ptr addrspace(67) %0, ptr addrspace(67) %1, i64 %n8, i1 true)
  call void @llvm.memmove.p67.p0.i32(ptr addrspace(67) %0, ptr %2, i32 %n4, i1 true)
  call void @llvm.memmove.inline.p0.p67.i32(ptr %2, ptr addrspace(67) %0, i32 %n4, i1 true)
  call void @llvm.memmove.inline.p67.p67.i64(ptr addrspace(67) %0, ptr addrspace(67) %1, i64 %n8, i1 true)
  call void @llvm.memmove.inline.p67.p0.i32(ptr addrspace(67) %0, ptr %2, i32 %n4, i1 true)

  call void @llvm.memset.p0.i32(ptr %2, i8 11, i32 %n4, i1 true)
  call void @llvm.memset.p67.i64(ptr addrspace(67) %1, i8 12, i64 %n8, i1 true)
  call void @llvm.memset.inline.p0.i32(ptr %2, i8 13, i32 %n4, i1 true)
  call void @llvm.memset.inline.p67.i64(ptr addrspace(67) %1, i8 14, i64 %n8, i1 true)

  ret void
}