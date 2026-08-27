; Check that calls to the kit.gpu.warp.size intrinsic are lowered correctly.
; On NVIDIA GPU's, the warp size is always 32.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt -passes=emb-lower-warp-intrinsics \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s

; CHECK-LABEL: @f
; CHECK-NEXT: ret i32 32
define i32 @f() {
  %1 = call i32 @llvm.kit.gpu.warp.size(i32 2)
  ret i32 %1
}
