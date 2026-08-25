; Check the Kitsune's warp.id intrinsics are lowered as expected.
;
; RUN: opt -passes=kit-lower-warp-intrinsics -S %s \
; RUN:     --tapir=hip --tapir-hip-arch=gfx1103 \
; RUN:     | FileCheck %s
;
; CHECK: define void @f() {
; CHECK-NEXT: call i32 @__kit_hip_warp_id(i32 1)
; CHECK-NEXT: call i32 @__kit_hip_warp_id(i32 2)
; CHECK-NEXT: call i32 @__kit_hip_warp_id(i32 3)
; CHECK-NEXT: ret void
;
; CHECK: define linkonce_odr i32 @__kit_hip_warp_id
; CHECK-SAME: i32 %[[DIMS:[^)]+]]
; CHECK-NEXT: %[[BSZX:.+]] = call i32 @llvm.kit.gpu.block.size.x(i32 4)
; CHECK-NEXT: %[[BSZY:.+]] = call i32 @llvm.kit.gpu.block.size.y(i32 4)
; CHECK-NEXT: %[[BSZXY:.+]] = mul i32 %[[BSZX]], %[[BSZY]]
; CHECK-NEXT: %[[TIDZ:.+]] = call i32 @llvm.kit.gpu.thread.id.z(i32 4)
; CHECK-NEXT: %[[OFFZ:.+]] = mul i32 %[[TIDZ]], %[[BSZXY]]
; CHECK-NEXT: %[[TIDY:.+]] = call i32 @llvm.kit.gpu.thread.id.y(i32 4)
; CHECK-NEXT: %[[OFFY:.+]] = mul i32 %[[TIDY]], %[[BSZX]]
; CHECK-NEXT: %[[X:.+]] = call i32 @llvm.kit.gpu.thread.id.x(i32 4)
; CHECK-NEXT: %[[HASY:.+]] = icmp ugt i32 %[[DIMS]], 1
; CHECK-NEXT: %[[Y:.+]] = select i1 %[[HASY]], i32 %[[OFFY]], i32 0
; CHECK-NEXT: %[[OFFXY:.+]] = add i32 %[[X]], %[[Y]]
; CHECK-NEXT: %[[HASZ:.+]] = icmp ugt i32 %[[DIMS]], 2
; CHECK-NEXT: %[[Z:.+]] = select i1 %[[HASZ]], i32 %[[OFFZ]], i32 0
; CHECK-NEXT: %[[OFFXYZ:.+]] = add i32 %[[OFFXY]], %[[Z]]
; CHECK-NEXT: %[[RESULT:.+]] = udiv i32 %[[OFFXYZ]], 32
; CHECK-NEXT: ret i32 %[[RESULT]]
;
; CHECK-NOT: define

define void @f() {
  %1 = call i32 @llvm.kit.gpu.warp.id(i32 4, i32 1)
  %2 = call i32 @llvm.kit.gpu.warp.id(i32 4, i32 2)
  %3 = call i32 @llvm.kit.gpu.warp.id(i32 4, i32 3)
  ret void
}
