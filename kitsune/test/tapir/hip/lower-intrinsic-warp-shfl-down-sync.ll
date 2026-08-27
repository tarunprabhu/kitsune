; Check the Kitsune's warp.shfl.down.sync intrinsics are lowered as expected.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt -passes='emb-lower-warp-intrinsics' \
; RUN:           --tapir=hip --tapir-hip-arch=gfx90a \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s

; CHECK-LABEL: @i32
; CHECK-SAME: i32 %[[VAL:[^,]+]]
; CHECK-SAME: i32 %[[OFFSET:[^)]+]]
; CHECK-NEXT: %[[RESULT:.+]] = call i32 @__kit.hip.warp.shfl.down.sync.i32
; CHECK-SAME: i32 %[[VAL]]
; CHECK-SAME: i32 %[[OFFSET]]
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @i32(i32 %val, i32 %offset) {
  %1 = call i32 @llvm.kit.gpu.warp.shfl.down.sync.i32(i32 4, i32 %val, i32 %offset)
  ret i32 %1
}

; CHECK-LABEL: @i64
; CHECK-SAME: i64 %[[VAL:[^,]+]]
; CHECK-SAME: i32 %[[OFFSET:[^)]+]]
; CHECK-NEXT: %[[RESULT:.+]] = call i64 @__kit.hip.warp.shfl.down.sync.i64
; CHECK-SAME: i64 %[[VAL]]
; CHECK-SAME: i32 %[[OFFSET]]
; CHECK-NEXT: ret i64 %[[RESULT]]
define i64 @i64(i64 %val, i32 %offset) {
  %1 = call i64 @llvm.kit.gpu.warp.shfl.down.sync.i64(i32 4, i64 %val, i32 %offset)
  ret i64 %1
}

; CHECK-LABEL: @f32
; CHECK-SAME: float %[[VAL:[^,]+]]
; CHECK-SAME: i32 %[[OFFSET:[^)]+]]
; CHECK-NEXT: %[[RESULT:.+]] = call float @__kit.hip.warp.shfl.down.sync.f32
; CHECK-SAME: float %[[VAL]]
; CHECK-SAME: i32 %[[OFFSET]]
; CHECK-NEXT: ret float %[[RESULT]]
define float @f32(float %val, i32 %offset) {
  %1 = call float @llvm.kit.gpu.warp.shfl.down.sync.f32(i32 4, float %val, i32 %offset)
  ret float %1
}

; CHECK-LABEL: @f64
; CHECK-SAME: double %[[VAL:[^,]+]]
; CHECK-SAME: i32 %[[OFFSET:[^)]+]]
; CHECK-NEXT: %[[RESULT:.+]] = call double @__kit.hip.warp.shfl.down.sync.f64
; CHECK-SAME: double %[[VAL]]
; CHECK-SAME: i32 %[[OFFSET]]
; CHECK-NEXT: ret double %[[RESULT]]
define double @f64(double %val, i32 %offset) {
  %1 = call double @llvm.kit.gpu.warp.shfl.down.sync.f64(i32 4, double %val, i32 %offset)
  ret double %1
}

; CHECK-LABEL: define linkonce_odr i32 @__kit.hip.lane.id
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[LANE32:.+]] = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i32 64, 32
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[BB64:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BB64]]:
; CHECK-NEXT: %[[LANE64:.+]] = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %[[LANE32]])
; CHECK-NEXT: br label %[[EXIT]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: %[[RESULT:.+]] = phi i32
; CHECK-SAME: [ %[[LANE32]]
; CHECK-SAME: [ %[[LANE64]], %[[BB64]] ]
; CHECK-NEXT: ret i32 %[[RESULT]]

; CHECK-LABEL: define linkonce_odr i32 @__kit.hip.warp.shfl.down.sync.core
; CHECK-SAME: i32 %[[VAL:[^,]+]]
; CHECK-SAME: i32 %[[OFFSET:[^)]+]]
; CHECK-SAME: #[[ATTRS:[0-9]+]]
; CHECK-NEXT: %[[ID:.+]] = call i32 @__kit.hip.lane.id()
; CHECK-NEXT: %[[MASK:.+]] = sub i32 64, 1
; CHECK-NEXT: %[[LANE:.+]] = and i32 %[[ID]], %[[MASK]]
; CHECK-NEXT: %[[NGBR:.+]] = add i32 %[[LANE]], %[[OFFSET]]
; CHECK-NEXT: %[[INDEX:.+]] = shl i32 %[[NGBR]], 2
; CHECK-NEXT: %[[RESULT:.+]] = call i32 @llvm.amdgcn.ds.bpermute
; CHECK-SAME: i32 %[[INDEX]]
; CHECK-SAME: i32 %[[VAL]]
; CHECK-NEXT: ret i32 %[[RESULT]]

; CHECK-LABEL: define linkonce_odr i32 @__kit.hip.warp.shfl.down.sync.i32
; CHECK-SAME: i32 %[[VAL:[^,]+]]
; CHECK-SAME: i32 %[[OFFSET:[^)]+]]
; CHECK-SAME: #[[ATTRS]]
; CHECK-NEXT: %[[RESULT:.+]] = call i32 @__kit.hip.warp.shfl.down.sync.core
; CHECK-SAME: i32 %[[VAL]]
; CHECK-SAME: i32 %[[OFFSET]]
; CHECK-NEXT: ret i32 %[[RESULT]]

; CHECK-LABEL: define linkonce_odr i64 @__kit.hip.warp.shfl.down.sync.i64
; CHECK-SAME: i64 %[[VAL:[^,]+]]
; CHECK-SAME: i32 %[[OFFSET:[^)]+]]
; CHECK-SAME: #[[ATTRS]]
; CHECK-NEXT: %[[L32:.+]] = trunc i64 %[[VAL]] to i32
; CHECK-NEXT: %[[RL32T:.+]] = call i32 @__kit.hip.warp.shfl.down.sync.core
; CHECK-SAME: i32 %[[L32]]
; CHECK-SAME: i32 %[[OFFSET]]
; CHECK-NEXT: %[[RL32:.+]] = zext i32 %[[RL32T]] to i64
; CHECK-NEXT: %[[SHU32:.+]] = lshr i64 %[[VAL]], 32
; CHECK-NEXT: %[[U32:.+]] = trunc nuw i64 %[[SHU32]] to i32
; CHECK-NEXT: %[[RU32T:.+]] = call i32 @__kit.hip.warp.shfl.down.sync.core
; CHECK-SAME: i32 %[[U32]]
; CHECK-SAME: i32 %[[OFFSET]]
; CHECK-NEXT: %[[ZU32:.+]] = zext i32 %[[RU32T]] to i64
; CHECK-NEXT: %[[RU32:.+]] = shl nsw i64 %[[ZU32]], 32
; CHECK-NEXT: %[[RES:.+]] = or disjoint i64 %[[RU32]], %[[RL32]]
; CHECK-NEXT: ret i64 %[[RES]]

; CHECK-LABEL: define linkonce_odr float @__kit.hip.warp.shfl.down.sync.f32
; CHECK-SAME: float %[[VAL:[^,]+]]
; CHECK-SAME: i32 %[[OFFSET:[^)]+]]
; CHECK-SAME: #[[ATTRS]]
; CHECK-NEXT: %[[CST:.+]] = bitcast float %[[VAL]] to i32
; CHECK-NEXT: %[[RI:.+]] = call i32 @__kit.hip.warp.shfl.down.sync.core
; CHECK-SAME: i32 %[[CST]]
; CHECK-SAME: i32 %[[OFFSET]]
; CHECK-NEXT: %[[RESULT:.+]] = bitcast i32 %[[RI]] to float
; CHECK-NEXT: ret float %[[RESULT]]

; CHECK-LABEL: define linkonce_odr double @__kit.hip.warp.shfl.down.sync.f64
; CHECK-SAME: double %[[VAL:[^,]+]]
; CHECK-SAME: i32 %[[OFFSET:[^)]+]]
; CHECK-SAME: #[[ATTRS]]
; CHECK-NEXT: %[[V64:.+]] = bitcast double %[[VAL]] to i64
; CHECK-NEXT: %[[L32:.+]] = trunc i64 %[[V64]] to i32
; CHECK-NEXT: %[[RL32T:.+]] = call i32 @__kit.hip.warp.shfl.down.sync.core
; CHECK-SAME: i32 %[[L32]]
; CHECK-SAME: i32 %[[OFFSET]]
; CHECK-NEXT: %[[RL32:.+]] = zext i32 %[[RL32T]] to i64
; CHECK-NEXT: %[[SHU32:.+]] = lshr i64 %[[V64]], 32
; CHECK-NEXT: %[[U32:.+]] = trunc nuw i64 %[[SHU32]] to i32
; CHECK-NEXT: %[[RU32T:.+]] = call i32 @__kit.hip.warp.shfl.down.sync.core
; CHECK-SAME: i32 %[[U32]]
; CHECK-SAME: i32 %[[OFFSET]]
; CHECK-NEXT: %[[ZU32:.+]] = zext i32 %[[RU32T]] to i64
; CHECK-NEXT: %[[RU32:.+]] = shl nsw i64 %[[ZU32]], 32
; CHECK-NEXT: %[[RES64:.+]] = or disjoint i64 %[[RU32]], %[[RL32]]
; CHECK-NEXT: %[[RES:.+]] = bitcast i64 %[[RES64]] to double
; CHECK-NEXT: ret double %[[RES]]

; CHECK: attributes #[[ATTRS]]
; CHECK-SAME: convergent
; CHECK-SAME: mustprogress
; CHECK-SAME: nofree
; CHECK-SAME: norecurse
; CHECK-SAME: nounwind
; CHECK-SAME: willreturn
; CHECK-SAME: memory(none)
