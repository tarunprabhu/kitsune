; Check that attributes are preserved when lowering the kit.async.launch.threads
; intrinsic.
;
; RUN: opt --tapir=pthreads -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: call ptr @__kitpthr_launch(ptr @f, i64 0, i64 %n, i64 1, ptr @gbuf){{$}}
; CHECK-NEXT: %[[CTX:[0-9]+]] = call ptr @__kitpthr_launch(ptr nonnull @f, i64 0, i64 %n, i64 1, ptr nonnull @gbuf) #[[LAUNCH:[0-9]+]]
; CHECK-NEXT: call void @__kitpthr_sync(ptr nonnull %[[CTX]]) #[[SYNC:[0-9]+]]
; CHECK-NEXT: ret void
;
; CHECK: attributes #[[LAUNCH]] = { "launch" }
; CHECK: attributes #[[SYNC]] = { "sync" }

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-unknown-linux-gnu"

@gbuf = external global [7 x float]

define void @f(i64 %n) {
  %1 = call ptr @llvm.kit.async.launch.threads(i32 1024, ptr @f, i64 0, i64 %n, i64 1, ptr @gbuf)
  %2 = call ptr @llvm.kit.async.launch.threads(i32 1024, ptr nonnull @f, i64 0, i64 %n, i64 1, ptr nonnull @gbuf) #0
  call void @llvm.kit.sync.threads(i32 1024, ptr nonnull %2) #1
  ret void
}

attributes #0 = { "launch" }
attributes #1 = { "sync" }
