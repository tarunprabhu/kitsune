; REQUIRES: kitsune-hip
;
; The Kitsune-specific GPU thread intrinsics should have been lowered by the
; emb-lower-intrinsics-gpu-thread-hip pass. A catastrophic failure is expected
; if these intrinsics are present when the codegen passes are run.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | not --crash \
; RUN:       opt --tapir=hip -passes='emb-lower-intrinsics' -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: GPU thread intrinsics in AMDGPU device modules should have been
; CHECK-SAME: replaced by the emb-lower-intrinsics-early pass
; CHECK: UNREACHABLE executed

declare void @ext(i32)

define void @f() {
threadidx:
  %tid.x = call i32 @llvm.kit.gpu.thread.id.x(i32 4);
  %tid.y = call i32 @llvm.kit.gpu.thread.id.y(i32 4);
  %tid.z = call i32 @llvm.kit.gpu.thread.id.z(i32 4);
  br label %blockidx

blockidx:
  %bid.x = call i32 @llvm.kit.gpu.block.id.x(i32 4);
  %bid.y = call i32 @llvm.kit.gpu.block.id.y(i32 4);
  %bid.z = call i32 @llvm.kit.gpu.block.id.z(i32 4);
  br label %blockdim

blockdim:
  %bsz.x = call i32 @llvm.kit.gpu.block.size.x(i32 4);
  %bsz.y = call i32 @llvm.kit.gpu.block.size.y(i32 4);
  %bsz.z = call i32 @llvm.kit.gpu.block.size.z(i32 4);
  br label %griddim

griddim:
  %gsz.x = call i32 @llvm.kit.gpu.grid.size.x(i32 4);
  %gsz.y = call i32 @llvm.kit.gpu.grid.size.y(i32 4);
  %gsz.z = call i32 @llvm.kit.gpu.grid.size.z(i32 4);
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
  ret void
}
