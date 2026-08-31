; Check the Kitsune's warp.id intrinsics are lowered as expected.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt -passes=emb-lower-intrinsics \
; RUN:           --tapir=hip --tapir-hip-arch=gfx1103 \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s
;
; CHECK: define void @f() {
; CHECK-NEXT: call i32 @__kit.hip.warp.id(i32 1)
; CHECK-NEXT: call i32 @__kit.hip.warp.id(i32 2)
; CHECK-NEXT: call i32 @__kit.hip.warp.id(i32 3)
; CHECK-NEXT: ret void
;
; CHECK: define linkonce_odr i32 @__kit.hip.warp.id
; CHECK-SAME: i32 %[[DIMS:[^)]+]]
; CHECK-SAME: #[[ATTRS:[0-9]+]]
; CHECK-NEXT: %[[BSZX64:.+]] = call i64 @__ockl_get_local_size(i32 0)
; CHECK-NEXT: %[[BSZX:.+]] = trunc i64 %[[BSZX64]] to i32
; CHECK-NEXT: %[[BSZY64:.+]] = call i64 @__ockl_get_local_size(i32 1)
; CHECK-NEXT: %[[BSZY:.+]] = trunc i64 %[[BSZY64]] to i32
; CHECK-NEXT: %[[BSZXY:.+]] = mul i32 %[[BSZX]], %[[BSZY]]
; CHECK-NEXT: %[[TIDZ:.+]] = call i32 @llvm.amdgcn.workitem.id.z()
; CHECK-NEXT: %[[OFFZ:.+]] = mul i32 %[[TIDZ]], %[[BSZXY]]
; CHECK-NEXT: %[[TIDY:.+]] = call i32 @llvm.amdgcn.workitem.id.y()
; CHECK-NEXT: %[[OFFY:.+]] = mul i32 %[[TIDY]], %[[BSZX]]
; CHECK-NEXT: %[[X:.+]] = call i32 @llvm.amdgcn.workitem.id.x()
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
;
; CHECK: attributes #[[ATTRS]]
; CHECK-SAME: convergent
; CHECK-SAME: mustprogress
; CHECK-SAME: nofree
; CHECK-SAME: norecurse
; CHECK-SAME: nounwind
; CHECK-SAME: willreturn
; CHECK-SAME: memory(none)

define void @f() {
  %1 = call i32 @llvm.kit.gpu.warp.id(i32 4, i32 1)
  %2 = call i32 @llvm.kit.gpu.warp.id(i32 4, i32 2)
  %3 = call i32 @llvm.kit.gpu.warp.id(i32 4, i32 3)
  ret void
}
