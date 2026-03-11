; Check that calls to printf are lowered correctly.
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:     --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:     -passes='loop-spawning,emb-resolve-libdevice-calls' -S %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: %[[TYPE:.+]] = type { i64, ptr }
;
; CHECK-DAG: @[[FMT:.+]] = internal constant [9 x i8] c"%ld: %s\0A\00"
; CHECK-DAG: @[[STR:.+]] = internal constant [12 x i8] c"Hello world\00"
;
; CHECK: %[[ARGS:.+]] = alloca %[[TYPE]]
; CHECK-DAG: %[[ARG0:.+]] = getelementptr %[[TYPE]], ptr %[[ARGS]], i64 0, i32 0
; CHECK-DAG: store i64 %{{.+}}, ptr %[[ARG0]]
; CHECK-DAG: %[[ARG1:.+]] = getelementptr %[[TYPE]], ptr %[[ARGS]], i64 0, i32 1
; CHECK-DAG: store ptr @[[STR]], ptr %[[ARG1]]
; CHECK: call i32 @vprintf(ptr @[[FMT]], ptr %[[ARGS]])
; CHECK-NOT: @printf

@.str = private unnamed_addr constant [9 x i8] c"%ld: %s\0A\00", align 1
@.str.1 = private unnamed_addr constant [12 x i8] c"Hello world\00", align 1

declare i32 @printf(ptr, ...)

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %call7 = call i32 (ptr, ...) @printf(ptr @.str, i64 %i, ptr @.str.1)
  reattach within %syncreg, label %latch

latch:
  %i.inc = add nuw nsw i64 %i, 1
  %exitcond.not = icmp eq i64 %i.inc, %n
  br i1 %exitcond.not, label %sync, label %header, !llvm.loop !0

sync:
  sync within %syncreg, label %forall.end

forall.end:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.lowering.enabled"}
