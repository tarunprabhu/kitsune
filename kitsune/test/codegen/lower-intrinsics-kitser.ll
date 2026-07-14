; Check that intrinsics that map to Kitsune's serial runtime are lowered
; correctly. If more intrinsics are created, they should be added here to test
; basic intrinsic lowering.
;
; RUN: opt --tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: call void @__kitser_initialize()
; CHECK-NEXT: call void @__kitser_finalize()

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

define void @f(ptr %buf, i64 %n) {
  call void @llvm.kit.runtime.initialize(i32 1)
  call void @llvm.kit.runtime.finalize(i32 1)
  ret void
}
