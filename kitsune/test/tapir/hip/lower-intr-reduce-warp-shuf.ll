; Check that the gpu.reduce.warp.shuffle intrinsic is lowered correctly.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt -passes=emb-lower-intrinsics \
; RUN:           --tapir=hip --tapir-hip-arch=gfx90a \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[RESULT:[^,]+]]
; CHECK-SAME: i64 %[[VAL:[^)]+]]
; CHECK-NEXT: %[[REDUCED:.+]] = call i64 @__kit.reduce.warp.shuffle.1.hip.umax.i64(i64 %[[VAL]])
; CHECK-NEXT: call void @__kit.reduce.warp.shuffle.final.1.hip.umax.i64
; CHECK-SAME: ptr %[[RESULT]]
; CHECK-SAME: i64 %[[REDUCED]]
; CHECK-NEXT: ret void
;
; CHECK-LABEL: define linkonce_odr i64 @__kit.reduce.warp.shuffle.1.hip.umax.i64
; CHECK-SAME: i64 %[[VAL:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[RES:.+]] = alloca i64
; CHECK-NEXT: store i64 %[[VAL]], ptr %[[RES]]
; CHECK-NEXT: %[[OFF0:.+]] = udiv i32 64, 2
; CHECK-NEXT: br label %[[BODY:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: %[[OFF:.+]] = phi i32
; CHECK-SAME: [ %[[OFF0]], %[[ENTRY]] ]
; CHECK-SAME: [ %[[OFFN:.+]], %[[BODY]] ]
; CHECK-NEXT: %[[CURR:.+]] = load i64, ptr %[[RES]]
; CHECK-NEXT: %[[NEW:.+]] = call i64 @__kit.hip.warp.shfl.down.sync.i64(i64 %[[CURR]], i32 %[[OFF]])
; CHECK-NEXT: call void @umax(ptr %[[RES]], i64 %[[NEW]])
; CHECK-NEXT: %[[OFFN]] = udiv i32 %[[OFF]], 2
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i32 %[[OFFN]], 0
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[BODY]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: %[[FINAL:.+]] = load i64, ptr %[[RES]]
; CHECK-NEXT: ret i64 %[[FINAL]]
;
; CHECK-LABEL: define linkonce_odr void @__kit.reduce.warp.shuffle.final.1.hip.umax.i64
; CHECK-SAME: ptr %[[RESULT:[^,]+]]
; CHECK-SAME: i64 %[[VAL:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[LANE:.+]] = call i32 @__kit.hip.warp.lane(i32 1)
; CHECK-NEXT: %[[IS0:.+]] = icmp eq i32 %[[LANE]], 0
; CHECK-NEXT: br i1 %[[IS0]], label %[[REDUCE:.+]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REDUCE]]:
; CHECK-NEXT: atomicrmw umax ptr %[[RESULT]], i64 %[[VAL]] monotonic
; CHECK-NEXT: br label %[[EXIT]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void

declare void @umax(ptr, i64)

define void @f(ptr %result, i64 %v) !kit.func !0 {
  call void(i32, i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.gpu.reduce.warp.shuffle(i32 4, i32 26, ptr %result, i32 8, i64 %v, i64 0, ptr @umax)
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"kit.func.kernel", i32 1}
