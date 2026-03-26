; Check that the command line option to inline all device functions is handled
; correctly.
;
; ------------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda -passes='loop-spawning,emb-prepare' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s -check-prefixes ALL,DEFAULT
;
; ALL-LABEL: define {{.+}} @sieve(
; ALL-SAME: #[[ATTRS_SIEVE:[0-9]+]]
; ALL-SAME: !kit.func ![[MD_SIEVE:[0-9]+]]
;
; ALL-LABEL: define {{.+}} @id(
; ALL-SAME: #[[ATTRS_ID:[0-9]+]]
; ALL-SAME: !kit.func ![[MD_ID:[0-9]+]]
;
; DEFAULT-DAG: attributes #[[ATTRS_SIEVE]] = { "
; DEFAULT-DAG: attributes #[[ATTRS_ID]] = { noinline "
;
; ALL-DAG: ![[DEVICE:[0-9]+]] = !{!"kit.func.device"}
; ALL-DAG: ![[MD_SIEVE]] = distinct !{![[MD_SIEVE]], ![[DEVICE]]}
; ALL-DAG: ![[MD_ID]] = distinct !{![[MD_ID]], ![[DEVICE]]}
;
; ------------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda -passes='loop-spawning,emb-prepare' %s \
; RUN:     -emb-inline-all \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s -check-prefixes ALL,INLINE
;
; INLINE-DAG: attributes #[[ATTRS_SIEVE]] = { alwaysinline "
; INLINE-DAG: attributes #[[ATTRS_ID]] = { noinline "
;
; ------------------------------------------------------------------------------

; Function Attrs: noinline
define i64 @id(i64 %n) #0 {
  ret i64 %n
}

define i64 @sieve(i64 %0) {
  %2 = alloca [256 x i8], align 16
  call void @llvm.lifetime.start.p0(i64 256, ptr nonnull %2)
  br label %5

3:
  %4 = icmp slt i64 %0, 4
  br i1 %4, label %10, label %16

5:
  %6 = phi i64 [ 0, %1 ], [ %8, %5 ]
  %7 = getelementptr [256 x i8], ptr %2, i64 0, i64 %6
  store i8 1, ptr %7, align 1
  %8 = add i64 %6, 1
  %9 = icmp eq i64 %8, 256
  br i1 %9, label %3, label %5, !llvm.loop !6

10:
  %11 = add i64 %0, -1
  %12 = getelementptr [256 x i8], ptr %2, i64 0, i64 %11
  %13 = load i8, ptr %12, align 1
  %14 = trunc i8 %13 to i1
  %15 = add i64 %0, -2
  br i1 %14, label %39, label %34

16:
  %17 = phi i64 [ %31, %30 ], [ 2, %3 ]
  %18 = phi i64 [ %32, %30 ], [ 4, %3 ]
  %19 = getelementptr [256 x i8], ptr %2, i64 0, i64 %17
  %20 = load i8, ptr %19, align 1
  %21 = trunc i8 %20 to i1
  %22 = and i64 %18, 4294967295
  %23 = icmp sle i64 %22, %0
  %24 = and i1 %23, %21
  br i1 %24, label %25, label %30

25:
  %26 = phi i64 [ %28, %25 ], [ %22, %16 ]
  %27 = getelementptr [256 x i8], ptr %2, i64 0, i64 %26
  store i8 0, ptr %27, align 1
  %28 = add i64 %26, %17
  %29 = icmp sgt i64 %28, %0
  br i1 %29, label %30, label %25, !llvm.loop !7

30:
  %31 = add i64 %17, 1
  %32 = mul i64 %31, %31
  %33 = icmp sgt i64 %32, %0
  br i1 %33, label %10, label %16, !llvm.loop !8

34:
  %35 = getelementptr [256 x i8], ptr %2, i64 0, i64 %15
  %36 = load i8, ptr %35, align 1
  %37 = zext nneg i8 %36 to i64
  %38 = add i64 %37, %0
  br label %39

39:
  %40 = phi i64 [ %38, %34 ], [ %15, %10 ]
  call void @llvm.lifetime.end.p0(i64 256, ptr nonnull %2)
  ret i64 %40
}

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr i32, ptr %c, i64 %i
  %.call1 = call i64 @sieve(i64 %n)
  %.call2 = call i64 @id(i64 %n)
  %.sum = add i64 %.call1, %.call2
  store i64 %.sum, ptr %arrayidx, align 4
  reattach within %syncreg, label %latch

latch:
  %i.next = add i64 %i, 1
  %cmp.i = icmp eq i64 %i.next, %n
  br i1 %cmp.i, label %sync, label %header, !llvm.loop !0

sync:
  sync within %syncreg, label %exit

exit:
  ret void
}

attributes #0 = { noinline }

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{!"tapir.loop.perfect.depth", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 1}
!6 = distinct !{!6}
!7 = distinct !{!7}
!8 = distinct !{!8}
