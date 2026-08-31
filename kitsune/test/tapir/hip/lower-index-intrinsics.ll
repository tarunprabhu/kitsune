; Check that the Kitsune-specific GPU index intrinsics in an embedded module
; are lowered correctly.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:      | opt -passes='emb-lower-intrinsics' \
; RUN:      | %kit-mbc -S \
; RUN:      | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: [[THREADIDX:.+]]:
; CHECK-NEXT: %[[TIDX:.+]] = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK-NEXT: %[[TIDY:.+]] = call i32 @llvm.amdgcn.workitem.id.y()
; CHECK-NEXT: %[[TIDZ:.+]] = call i32 @llvm.amdgcn.workitem.id.z()
; CHECK-NEXT: br label %[[BLOCKIDX:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BLOCKIDX]]:
; CHECK-NEXT: %[[BIDX:.+]] = call i32 @llvm.amdgcn.workgroup.id.x()
; CHECK-NEXT: %[[BIDY:.+]] = call i32 @llvm.amdgcn.workgroup.id.y()
; CHECK-NEXT: %[[BIDZ:.+]] = call i32 @llvm.amdgcn.workgroup.id.z()
; CHECK-NEXT: br label %[[BLOCKDIM:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BLOCKDIM]]:
; CHECK-NEXT: %[[BSZX64:.+]] = call i64 @__ockl_get_local_size(i32 0)
; CHECK-NEXT: %[[BSZX:.+]] = trunc i64 %[[BSZX64]] to i32
; CHECK-NEXT: %[[BSZY64:.+]] = call i64 @__ockl_get_local_size(i32 1)
; CHECK-NEXT: %[[BSZY:.+]] = trunc i64 %[[BSZY64]] to i32
; CHECK-NEXT: %[[BSZZ64:.+]] = call i64 @__ockl_get_local_size(i32 2)
; CHECK-NEXT: %[[BSZZ:.+]] = trunc i64 %[[BSZZ64]] to i32
; CHECK-NEXT: br label %[[GRIDDIM:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[GRIDDIM]]:
; CHECK-NEXT: %[[GSZX64:.+]] = call i64 @__ockl_get_global_size(i32 0)
; CHECK-NEXT: %[[GSZX:.+]] = trunc i64 %[[GSZX64]] to i32
; CHECK-NEXT: %[[GSZY64:.+]] = call i64 @__ockl_get_global_size(i32 1)
; CHECK-NEXT: %[[GSZY:.+]] = trunc i64 %[[GSZY64]] to i32
; CHECK-NEXT: %[[GSZZ64:.+]] = call i64 @__ockl_get_global_size(i32 2)
; CHECK-NEXT: %[[GSZZ:.+]] = trunc i64 %[[GSZZ64]] to i32
; CHECK-NEXT: br label %[[USES:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[USES]]:
; CHECK-NEXT: call void @ext(i32 %[[TIDX]])
; CHECK-NEXT: call void @ext(i32 %[[TIDY]])
; CHECK-NEXT: call void @ext(i32 %[[TIDZ]])
; CHECK-NEXT: call void @ext(i32 %[[BIDX]])
; CHECK-NEXT: call void @ext(i32 %[[BIDY]])
; CHECK-NEXT: call void @ext(i32 %[[BIDZ]])
; CHECK-NEXT: call void @ext(i32 %[[BSZX]])
; CHECK-NEXT: call void @ext(i32 %[[BSZY]])
; CHECK-NEXT: call void @ext(i32 %[[BSZZ]])
; CHECK-NEXT: call void @ext(i32 %[[GSZX]])
; CHECK-NEXT: call void @ext(i32 %[[GSZY]])
; CHECK-NEXT: call void @ext(i32 %[[GSZZ]])
; CHECK-NEXT: ret void

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
