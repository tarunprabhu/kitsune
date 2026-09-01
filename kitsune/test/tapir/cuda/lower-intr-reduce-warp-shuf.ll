; Check that the gpu.reduce.warp.shuffle intrinsic is lowered correctly.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt -passes=emb-lower-intrinsics \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[RESULT:[^,]+]]
; CHECK-SAME: i32 %[[VAL:[^)]+]]
; CHECK-NEXT: %[[REDUCED:.+]] = call i32 @__kit.reduce.warp.shuffle.1.cuda.add.i32(i32 %[[VAL]])
; CHECK-NEXT: call void @__kit.reduce.warp.shuffle.final.1.cuda.add.i32
; CHECK-SAME: ptr %[[RESULT]]
; CHECK-SAME: i32 %[[REDUCED]]
; CHECK-NEXT: ret void
;
; CHECK-LABEL: define linkonce_odr i32 @__kit.reduce.warp.shuffle.1.cuda.add.i32
; CHECK-SAME: i32 %[[VAL:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[RES:.+]] = alloca i32
; CHECK-NEXT: store i32 %[[VAL]], ptr %[[RES]]
; CHECK-NEXT: %[[OFF0:.+]] = udiv i32 32, 2
; CHECK-NEXT: br label %[[BODY:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: %[[OFF:.+]] = phi i32
; CHECK-SAME: [ %[[OFF0]], %[[ENTRY]] ]
; CHECK-SAME: [ %[[OFFN:.+]], %[[BODY]] ]
; CHECK-NEXT: %[[CURR:.+]] = load i32, ptr %[[RES]]
; CHECK-NEXT: %[[NEW:.+]] = call i32 @__kit.cuda.warp.shfl.down.sync.i32(i32 %[[CURR]], i32 %[[OFF]])
; CHECK-NEXT: call void @sum(ptr %[[RES]], i32 %[[NEW]])
; CHECK-NEXT: %[[OFFN]] = udiv i32 %[[OFF]], 2
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i32 %[[OFFN]], 0
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[BODY]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: %[[FINAL:.+]] = load i32, ptr %[[RES]]
; CHECK-NEXT: ret i32 %[[FINAL]]
;
; CHECK-LABEL: @__kit.reduce.warp.shuffle.final.1.cuda.add.i32
; CHECK-SAME: ptr %[[RESULT:[^,]+]]
; CHECK-SAME: i32 %[[VAL:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[LANE:.+]] = call i32 @__kit.cuda.warp.lane(i32 1)
; CHECK-NEXT: %[[IS0:.+]] = icmp eq i32 %[[LANE]], 0
; CHECK-NEXT: br i1 %[[IS0]], label %[[REDUCE:.+]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REDUCE]]:
; CHECK-NEXT: atomicrmw add ptr %[[RESULT]], i32 %[[VAL]] monotonic
; CHECK-NEXT: br label %[[EXIT]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void

declare void @sum(ptr, i32)

define void @f(ptr %result, i32 %v) !kit.func !0 {
  call void(i32, i32, ptr, i32, i32, i32, ptr, ...) @llvm.kit.gpu.reduce.warp.shuffle(i32 2, i32 5, ptr %result, i32 4, i32 %v, i32 0, ptr @sum)
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"kit.func.kernel", i32 1}
