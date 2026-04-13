; Check that intrinsics that map to Kitsune's openmp runtime are lowered
; correctly. If more intrinsics are created, they should be added here to test
; basic intrinsic lowering.
;
; RUN: opt --tapir=openmp -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: call void @__kitomp_initialize()
; CHECK-NEXT: call void @__kitomp_launch(ptr @f, i64 0, i64 128, i64 1, ptr @gbuf)
; CHECK-NEXT: call void @__kitomp_finalize()

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

@gbuf = external global [7 x float]

define void @f(ptr %buf, i64 %n) {
  call void @llvm.kit.initialize(i32 512)
  call void @llvm.kit.launch.threads(i32 512, ptr @f, i64 0, i64 128, i64 1, ptr @gbuf)
  call void @llvm.kit.finalize(i32 512)
  ret void
}
