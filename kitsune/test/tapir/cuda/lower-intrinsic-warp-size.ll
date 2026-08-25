; Check that calls to the kit.gpu.warp.size intrinsic are lowered correctly.
; On NVIDIA GPU's, the warp size is always 32.
;
; RUN: opt -passes=kit-lower-warp-intrinsics -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK-NEXT: ret i32 32
define i32 @f() {
  %1 = call i32 @llvm.kit.gpu.warp.size(i32 2)
  ret i32 %1
}
