; REQUIRES: kitsune-hip
;
; Check that the Kitsune-specific intriniscs in an embedded module that are to
; be lowered to libdevice functions are lowered correctly.
;
; This test intentionally includes tests calls to intrinsics that are not
; lowered. Those intrinsics that can be lowered to existing LLVM intrinsics are
; lowered as part of codegen.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:      | opt --tapir=hip -passes='emb-lower-intrinsics-libdevice' \
; RUN:      | %kit-mbc --tapir=hip -S \
; RUN:      | FileCheck %s
;
; CHECK: entry:
; CHECK-NEXT: %tid.x = call i32 @llvm.kit.gpu.thread.id.x()
; CHECK-NEXT: %tid.y = call i32 @llvm.kit.gpu.thread.id.y()
; CHECK-NEXT: %tid.z = call i32 @llvm.kit.gpu.thread.id.z()
; CHECK-NEXT: %bid.x = call i32 @llvm.kit.gpu.block.id.x()
; CHECK-NEXT: %bid.y = call i32 @llvm.kit.gpu.block.id.y()
; CHECK-NEXT: %bid.z = call i32 @llvm.kit.gpu.block.id.z()
; CHECK-NEXT: br label %changed
; CHECK-EMPTY:
; CHECK-NEXT: changed:
; CHECK-NEXT: %bsz.x.64 = call i64 @__ockl_get_local_size(i32 0)
; CHECK-NEXT: %bsz.x = trunc i64 %bsz.x.64 to i32
; CHECK-NEXT: %bsz.y.64 = call i64 @__ockl_get_local_size(i32 1)
; CHECK-NEXT: %bsz.y = trunc i64 %bsz.y.64 to i32
; CHECK-NEXT: %bsz.z.64 = call i64 @__ockl_get_local_size(i32 2)
; CHECK-NEXT: %bsz.z = trunc i64 %bsz.z.64 to i32
; CHECK-NEXT: %gsz.x.64 = call i64 @__ockl_get_global_size(i32 0)
; CHECK-NEXT: %gsz.x = trunc i64 %gsz.x.64 to i32
; CHECK-NEXT: %gsz.y.64 = call i64 @__ockl_get_global_size(i32 1)
; CHECK-NEXT: %gsz.y = trunc i64 %gsz.y.64 to i32
; CHECK-NEXT: %gsz.z.64 = call i64 @__ockl_get_global_size(i32 2)
; CHECK-NEXT: %gsz.z = trunc i64 %gsz.z.64 to i32
; CHECK-NEXT: br label %uses
; CHECK-EMPTY:
; CHECK-NEXT: uses:
; CHECK-NEXT: call void @ext(i32 %tid.x)
; CHECK-NEXT: call void @ext(i32 %tid.y)
; CHECK-NEXT: call void @ext(i32 %tid.z)
; CHECK-NEXT: call void @ext(i32 %bid.x)
; CHECK-NEXT: call void @ext(i32 %bid.y)
; CHECK-NEXT: call void @ext(i32 %bid.z)
; CHECK-NEXT: call void @ext(i32 %bsz.x)
; CHECK-NEXT: call void @ext(i32 %bsz.y)
; CHECK-NEXT: call void @ext(i32 %bsz.z)
; CHECK-NEXT: call void @ext(i32 %gsz.x)
; CHECK-NEXT: call void @ext(i32 %gsz.y)
; CHECK-NEXT: call void @ext(i32 %gsz.z)
; CHECK-NEXT: br label %exit
; CHECK-EMPTY:
; CHECK-NEXT: exit:
; CHECK-NEXT: ret void

declare void @ext(i32)

define void @f() {
entry:
  %tid.x = call i32 @llvm.kit.gpu.thread.id.x();
  %tid.y = call i32 @llvm.kit.gpu.thread.id.y();
  %tid.z = call i32 @llvm.kit.gpu.thread.id.z();
  %bid.x = call i32 @llvm.kit.gpu.block.id.x();
  %bid.y = call i32 @llvm.kit.gpu.block.id.y();
  %bid.z = call i32 @llvm.kit.gpu.block.id.z();
  br label %changed

changed:
  %bsz.x = call i32 @llvm.kit.gpu.block.size.x();
  %bsz.y = call i32 @llvm.kit.gpu.block.size.y();
  %bsz.z = call i32 @llvm.kit.gpu.block.size.z();
  %gsz.x = call i32 @llvm.kit.gpu.grid.size.x();
  %gsz.y = call i32 @llvm.kit.gpu.grid.size.y();
  %gsz.z = call i32 @llvm.kit.gpu.grid.size.z();
  br label %uses

uses:
  call void @ext(i32 %tid.x)
  call void @ext(i32 %tid.y)
  call void @ext(i32 %tid.z)
  call void @ext(i32 %bid.x)
  call void @ext(i32 %bid.y)
  call void @ext(i32 %bid.z)
  call void @ext(i32 %bsz.x)
  call void @ext(i32 %bsz.y)
  call void @ext(i32 %bsz.z)
  call void @ext(i32 %gsz.x)
  call void @ext(i32 %gsz.y)
  call void @ext(i32 %gsz.z)
  br label %exit

exit:
  ret void
}
