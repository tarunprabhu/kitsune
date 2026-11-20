; Check that calls to the kit.gpu.warp.size intrinsic are lowered correctly.
;
; Dealing with the warp size on AMDGPU is tricky. Some devices only support a
; warp size of 32, others only support 64. But a few support both. Which one is
; used is determined by either the target features set on the function
; containing the intrinsic call.
;
;   - If the target features have been set, they are always used.
;
;   - If the target features are not set, the device architecture is queried to
;     determine the warp size.
;
; If the device architecture is not set correctly, an error is raised. We do not
; check for the error case here - it is unlikely to happen in practice because
; the frontend will generally have set the target features and architecture.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt -passes=emb-lower-intrinsics \
; RUN:           --tapir=hip --tapir-hip-arch=gfx1103 \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s --check-prefixes=CHECK,WARP32
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt -passes=emb-lower-intrinsics \
; RUN:           --tapir=hip --tapir-hip-arch=gfx90a \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s --check-prefixes=CHECK,WARP64

; This function has +wavefrontsize32 in the target features. This should
; override anything on the command-line
;
; CHECK-LABEL: @warpSize
; WARP32-NEXT: ret i32 32
define i32 @warpSize32() #0 {
  %1 = call i32 @llvm.kit.gpu.warp.size(i32 4)
  ret i32 %1
}

; This function has +wavefrontsize64 in the target features. This should
; override anything on the command-line
;
; CHECK-LABEL: @warpSize
; WARP64-NEXT: ret i32 64
define i32 @warpSize64() #1 {
  %1 = call i32 @llvm.kit.gpu.warp.size(i32 4)
  ret i32 %1
}

; This function does not have any target features. The warp size that is used
; will be determined by the device architecture.
;
; CHECK-LABEL: @warpSize
; WARP32-NEXT: ret i32 32
; WARP64-NEXT: ret i32 64
define i32 @warpSize() {
  %1 = call i32 @llvm.kit.gpu.warp.size(i32 4)
  ret i32 %1
}

attributes #0 = { "target-features"="+dl-insts+wavefrontsize32" }
attributes #1 = { "target-features"="+wavefrontsize64" }
