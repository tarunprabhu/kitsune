; Check that attributes are preserved when lowering the kit.launch.threads
; intrinsic.
;
; RUN: opt --tapir=openmp -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: call void @__kitomp_launch(ptr @f, i64 0, i64 %n, i64 1, ptr @gbuf)
; CHECK-NEXT: call void @__kitomp_launch(ptr nonnull @f, i64 0, i64 %n, i64 1, ptr nonnull @gbuf) #[[LAUNCH:[0-9]+]]
; CHECK-NEXT: ret void
;
; CHECK: attributes #[[LAUNCH]] = { "launch" }

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

@gbuf = external global [7 x float]

define void @f(i64 %n) {
  call void @llvm.kit.cpu.threads.launch(i32 512, ptr @f, i64 0, i64 %n, i64 1, ptr @gbuf)
  call void @llvm.kit.cpu.threads.launch(i32 512, ptr nonnull @f, i64 0, i64 %n, i64 1, ptr nonnull @gbuf) #0
  ret void
}

attributes #0 = { "launch" }
