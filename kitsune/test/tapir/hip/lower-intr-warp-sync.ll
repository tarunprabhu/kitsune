; Check that the @llvm.kit.warp.sync intrinsic is lowered correctly.
;
; RUN: %kit-enc %s \
; RUN:     | opt -passes='emb-lower-intrinsics' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: fence syncscope("wavefront") release
; CHECK-NEXT: call void @llvm.amdgcn.wave.barrier()
; CHECK-NEXT: fence syncscope("wavefront") acquire
; CHECK-NEXT: ret void

define void @f() {
  call void @llvm.kit.gpu.warp.sync(i32 4)
  ret void
}
