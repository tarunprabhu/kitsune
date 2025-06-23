; RUN: opt -passes='lower-kitsune-runtime-intrinsics' -S %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.kitrt.enable.verbose(i8)

; Function Attrs: nounwind memory(inaccessiblemem: readwrite) uwtable
define dso_local void @f(ptr noundef %buf, i64 noundef %n) local_unnamed_addr #0 {
entry:
  call void @llvm.kitrt.enable.verbose(i8 1)
  call void @llvm.kitrt.enable.verbose(i8 0)
  ret void
}

attributes #0 = { nounwind uwtable }

; CHECK-LABEL: @f
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @__kitrt_enable_verbose_mode()
; CHECK-NOT: call void @__kitrt_enable_verbose_mode()
; CHECK-NEXT: ret void