; Check the Kitsune's warp index and lane intrinsics get optimized as expected.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt -passes='emb-lower-intrinsics,emb-optimize' -emb-O2 \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s

; CHECK-LABEL: @id1
; CHECK-NEXT: %[[X:.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NEXT: %[[RESULT:.+]] = lshr i32 %[[X]], 5
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @id1() {
  %1 = call i32 @llvm.kit.gpu.warp.id(i32 2, i32 1)
  ret i32 %1
}

; CHECK-LABEL: @id2
; CHECK-NEXT: %[[BSZX:.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
; CHECK-NEXT: %[[TIDY:.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
; CHECK-NEXT: %[[Y:.+]] = mul nuw nsw i32 %[[TIDY]], %[[BSZX]]
; CHECK-NEXT: %[[X:.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NEXT: %[[OFFXY:.+]] = add nuw nsw i32 %[[Y]], %[[X]]
; CHECK-NEXT: %[[RESULT:.+]] = lshr i32 %[[OFFXY]], 5
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @id2() {
  %1 = call i32 @llvm.kit.gpu.warp.id(i32 2, i32 2)
  ret i32 %1
}

; CHECK-LABEL: @id3
; CHECK-NEXT: %[[BSZX:.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
; CHECK-NEXT: %[[BSZY:.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
; CHECK-NEXT: %[[TIDZ:.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
; CHECK-NEXT: %[[Z:.+]] = mul nuw nsw i32 %[[TIDZ]], %[[BSZY]]
; CHECK-NEXT: %[[TIDY:.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
; CHECK-NEXT: %[[X:.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NEXT: %[[ADD:.+]] = add nuw nsw i32 %[[Z]], %[[TIDY]]
; CHECK-NEXT: %[[MUL:.+]] = mul nuw nsw i32 %[[ADD]], %[[BSZX]]
; CHECK-NEXT: %[[OFFXYZ:.+]] = add nuw nsw i32 %[[MUL]], %[[X]]
; CHECK-NEXT: %[[RESULT:.+]] = lshr i32 %[[OFFXYZ]], 5
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @id3() {
  %1 = call i32 @llvm.kit.gpu.warp.id(i32 2, i32 3)
  ret i32 %1
}

; CHECK-LABEL: @lane1
; CHECK-NEXT: %[[X:.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NEXT: %[[RESULT:.+]] = and i32 %[[X]], 31
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @lane1() {
  %1 = call i32 @llvm.kit.gpu.warp.lane(i32 2, i32 1)
  ret i32 %1
}

; CHECK-LABEL: @lane2
; CHECK-NEXT: %[[BSZX:.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
; CHECK-NEXT: %[[TIDY:.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
; CHECK-NEXT: %[[Y:.+]] = mul nuw nsw i32 %[[TIDY]], %[[BSZX]]
; CHECK-NEXT: %[[X:.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NEXT: %[[OFFXY:.+]] = add nuw nsw i32 %[[Y]], %[[X]]
; CHECK-NEXT: %[[RESULT:.+]] = and i32 %[[OFFXY]], 31
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @lane2() {
  %1 = call i32 @llvm.kit.gpu.warp.lane(i32 2, i32 2)
  ret i32 %1
}

; CHECK-LABEL: @lane3
; CHECK-NEXT: %[[BSZX:.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
; CHECK-NEXT: %[[BSZY:.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
; CHECK-NEXT: %[[TIDZ:.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
; CHECK-NEXT: %[[Z:.+]] = mul nuw nsw i32 %[[TIDZ]], %[[BSZY]]
; CHECK-NEXT: %[[TIDY:.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
; CHECK-NEXT: %[[X:.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NEXT: %[[ADD:.+]] = add nuw nsw i32 %[[Z]], %[[TIDY]]
; CHECK-NEXT: %[[MUL:.+]] = mul{{.*}} i32 %[[ADD]], %[[BSZX]]
; CHECK-NEXT: %[[OFFXYZ:.+]] = add nuw nsw i32 %[[MUL]], %[[X]]
; CHECK-NEXT: %[[RESULT:.+]] = and i32 %[[OFFXYZ]], 31
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @lane3() {
  %1 = call i32 @llvm.kit.gpu.warp.lane(i32 2, i32 3)
  ret i32 %1
}
