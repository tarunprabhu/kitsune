; The number of dimensions given to Kitsune's warp lane intrinsic must be in the
; range [1,3].
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: Dimensions in GPU intrinsic call must be in range [1,3]. Got '0'
; CHECK: Dimensions in GPU intrinsic call must be in range [1,3]. Got '4'

define void @f() {
  %1 = call i32 @llvm.kit.gpu.warp.lane(i32 2, i32 0)
  %2 = call i32 @llvm.kit.gpu.warp.lane(i32 4, i32 4)
  ret void
}
