; Check that allocas in the embedded bitcode that are not in the function entry
; block are hoisted to it.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt --tapir=cuda -passes=emb-hoist-allocas \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s

; If there are no allocas, this pass should have no effect.
;
; CHECK-LABEL: @none
; CHECK-NEXT: ret void
define void @none() {
  ret void
}

; If there are allocas in the entry block, only those after the first non-alloca
; should be moved.
;
; CHECK-LABEL: @entry_only
; CHECK-NEXT: %first = alloca i8
; CHECK-NEXT: %second = alloca i16
; CHECK-NEXT: %third = alloca i32
; CHECK-NEXT: call void @ext(ptr %second)
; CHECK-NEXT: br label %exit
define void @entry_only() {
  %first = alloca i8
  %second = alloca i16
  call void @ext(ptr %second)
  %third = alloca i32
  br label %exit

exit:
  ret void
}

; Any allocas that are moved, are moved before the first non-alloca in the
; entry block. Their relative and names should be preserved.
;
; CHECK-LABEL: @move
; CHECK-NEXT: %existing = alloca i32
; CHECK-NEXT: %move1 = alloca i32
; CHECK-NEXT: %move2 = alloca i32
; CHECK-NEXT: br label %bb1
; CHECK-EMPTY:
; CHECK-NEXT: bb1:
; CHECK-NEXT: call void @ext2(ptr %move1)
; CHECK-NEXT: br label %bb2
; CHECK-EMPTY:
; CHECK-NEXT: bb2:
; CHECK-NEXT: call void @ext2(ptr %move1)
; CHECK-NEXT: call void @ext1(ptr %move2)
; CHECK-NEXT: br label %exit
; CHECK-EMPTY:
; CHECK-NEXT: exit:
; CHECK-NEXT: ret void
define void @move() {
   %existing = alloca i32
   br label %bb1

bb1:
  %move1 = alloca i32
  call void @ext2(ptr %move1)
  br label %bb2

bb2:
  %move2 = alloca i32
  call void @ext2(ptr %move1)
  call void @ext1(ptr %move2)
  br label %exit

exit:
  ret void
}

; Allocas inside loops will also be moved
;
; CHECK-LABEL: @in_loop
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %from_loop = alloca double
; CHECK-NEXT: br label %[[LOOP:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOP]]:
; CHECK-NEXT: %[[IV:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-SAME: [ %[[INC:.+]], %[[LOOP]] ]
; CHECK-NEXT: %[[INC]] = add i64 %[[IV]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[LOOP]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
define void @in_loop(i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %from_loop = alloca double
  %inc = add i64 %iv, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

; If any allocas have operands that are instructions, they are not moved at all
;
; CHECK-LABEL: @dependent
; CHECK-SAME: i32 %[[N:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[CST:.+]] = zext i32 %[[N]] to i64
; CHECK-NEXT: alloca i32, i64 %[[CST]]
; CHECK-NEXT: br label %[[EXIT:.+]]
define void @dependent(i32 %n) {
entry:
  %1 = zext i32 %n to i64
  %2 = alloca i32, i64 %1
  br label %exit

exit:
  ret void
}

declare void @ext(ptr)
declare void @ext1(ptr)
declare void @ext2(ptr)

!0 = distinct !{!0}
