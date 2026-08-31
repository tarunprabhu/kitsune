; Check the Kitsune's warp index and lane intrinsics get optimized as expected.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt -passes='emb-lower-intrinsics,emb-optimize' -emb-O2 \
; RUN:           --tapir=hip --tapir-hip-arch=gfx1103 \
; RUN:           --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s

; CHECK-LABEL: @id1
; CHECK-NEXT: %[[BSZX64:.+]] = tail call i64 @__ockl_get_local_size(i32 0)
; CHECK-NEXT: %[[BSZY64:.+]] = tail call i64 @__ockl_get_local_size(i32 1)
; CHECK-NEXT: %[[X:.+]] = tail call i32 @llvm.amdgcn.workitem.id.x()
; CHECK-NEXT: %[[RESULT:.+]] = lshr i32 %[[X]], 5
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @id1() {
  %1 = call i32 @llvm.kit.gpu.warp.id(i32 4, i32 1)
  ret i32 %1
}

; CHECK-LABEL: @id2
; CHECK-NEXT: %[[BSZX64:.+]] = tail call i64 @__ockl_get_local_size(i32 0)
; CHECK-NEXT: %[[BSZX:.+]] = trunc i64 %[[BSZX64]] to i32
; CHECK-NEXT: %[[BSZY64:.+]] = tail call i64 @__ockl_get_local_size(i32 1)
; CHECK-NEXT: %[[TIDY:.+]] = tail call i32 @llvm.amdgcn.workitem.id.y()
; CHECK-NEXT: %[[Y:.+]] = mul i32 %[[TIDY]], %[[BSZX]]
; CHECK-NEXT: %[[X:.+]] = tail call i32 @llvm.amdgcn.workitem.id.x()
; CHECK-NEXT: %[[OFFXY:.+]] = add i32 %[[Y]], %[[X]]
; CHECK-NEXT: %[[RESULT:.+]] = lshr i32 %[[OFFXY]], 5
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @id2() {
  %1 = call i32 @llvm.kit.gpu.warp.id(i32 4, i32 2)
  ret i32 %1
}

; CHECK-LABEL: @id3
; CHECK-NEXT: %[[BSZX64:.+]] = tail call i64 @__ockl_get_local_size(i32 0)
; CHECK-NEXT: %[[BSZX:.+]] = trunc i64 %[[BSZX64]] to i32
; CHECK-NEXT: %[[BSZY64:.+]] = tail call i64 @__ockl_get_local_size(i32 1)
; CHECK-NEXT: %[[BSZY:.+]] = trunc i64 %[[BSZY64]] to i32
; CHECK-NEXT: %[[TIDZ:.+]] = tail call i32 @llvm.amdgcn.workitem.id.z()
; CHECK-NEXT: %[[Z:.+]] = mul i32 %[[TIDZ]], %[[BSZY]]
; CHECK-NEXT: %[[TIDY:.+]] = tail call i32 @llvm.amdgcn.workitem.id.y()
; CHECK-NEXT: %[[X:.+]] = tail call i32 @llvm.amdgcn.workitem.id.x()
; CHECK-NEXT: %[[ADD:.+]] = add i32 %[[Z]], %[[TIDY]]
; CHECK-NEXT: %[[MUL:.+]] = mul i32 %[[ADD]], %[[BSZX]]
; CHECK-NEXT: %[[OFFXYZ:.+]] = add i32 %[[MUL]], %[[X]]
; CHECK-NEXT: %[[RESULT:.+]] = lshr i32 %[[OFFXYZ]], 5
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @id3() {
  %1 = call i32 @llvm.kit.gpu.warp.id(i32 4, i32 3)
  ret i32 %1
}

; CHECK-LABEL: @lane1
; CHECK-NEXT: %[[BSZX64:.+]] = tail call i64 @__ockl_get_local_size(i32 0)
; CHECK-NEXT: %[[BSZY64:.+]] = tail call i64 @__ockl_get_local_size(i32 1)
; CHECK-NEXT: %[[X:.+]] = tail call i32 @llvm.amdgcn.workitem.id.x()
; CHECK-NEXT: %[[RESULT:.+]] = and i32 %[[X]], 31
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @lane1() {
  %1 = call i32 @llvm.kit.gpu.warp.lane(i32 4, i32 1)
  ret i32 %1
}

; CHECK-LABEL: @lane2
; CHECK-NEXT: %[[BSZX64:.+]] = tail call i64 @__ockl_get_local_size(i32 0)
; CHECK-NEXT: %[[BSZX:.+]] = trunc i64 %[[BSZX64]] to i32
; CHECK-NEXT: %[[BSZY64:.+]] = tail call i64 @__ockl_get_local_size(i32 1)
; CHECK-NEXT: %[[TIDY:.+]] = tail call i32 @llvm.amdgcn.workitem.id.y()
; CHECK-NEXT: %[[Y:.+]] = mul i32 %[[TIDY]], %[[BSZX]]
; CHECK-NEXT: %[[X:.+]] = tail call i32 @llvm.amdgcn.workitem.id.x()
; CHECK-NEXT: %[[OFFXY:.+]] = add i32 %[[Y]], %[[X]]
; CHECK-NEXT: %[[RESULT:.+]] = and i32 %[[OFFXY]], 31
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @lane2() {
  %1 = call i32 @llvm.kit.gpu.warp.lane(i32 4, i32 2)
  ret i32 %1
}

; CHECK-LABEL: @lane3
; CHECK-NEXT: %[[BSZX64:.+]] = tail call i64 @__ockl_get_local_size(i32 0)
; CHECK-NEXT: %[[BSZX:.+]] = trunc i64 %[[BSZX64]] to i32
; CHECK-NEXT: %[[BSZY64:.+]] = tail call i64 @__ockl_get_local_size(i32 1)
; CHECK-NEXT: %[[BSZY:.+]] = trunc i64 %[[BSZY64]] to i32
; CHECK-NEXT: %[[TIDZ:.+]] = tail call i32 @llvm.amdgcn.workitem.id.z()
; CHECK-NEXT: %[[Z:.+]] = mul i32 %[[TIDZ]], %[[BSZY]]
; CHECK-NEXT: %[[TIDY:.+]] = tail call i32 @llvm.amdgcn.workitem.id.y()
; CHECK-NEXT: %[[X:.+]] = tail call i32 @llvm.amdgcn.workitem.id.x()
; CHECK-NEXT: %[[ADD:.+]] = add i32 %[[Z]], %[[TIDY]]
; CHECK-NEXT: %[[MUL:.+]] = mul i32 %[[ADD]], %[[BSZX]]
; CHECK-NEXT: %[[OFFXYZ:.+]] = add i32 %[[MUL]], %[[X]]
; CHECK-NEXT: %[[RESULT:.+]] = and i32 %[[OFFXYZ]], 31
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @lane3() {
  %1 = call i32 @llvm.kit.gpu.warp.lane(i32 4, i32 3)
  ret i32 %1
}
