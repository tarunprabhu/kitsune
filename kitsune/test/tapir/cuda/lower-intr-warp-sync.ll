; Check that the @llvm.kit.warp.sync intrinsic is lowered correctly.
;
; RUN: %kit-enc %s \
; RUN:     | opt -passes='emb-lower-intrinsics' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: call void @llvm.nvvm.bar.warp.sync(i32 -1)
; CHECK-NEXT: ret void

define void @f() {
  call void @llvm.kit.gpu.warp.sync(i32 2)
  ret void
}
