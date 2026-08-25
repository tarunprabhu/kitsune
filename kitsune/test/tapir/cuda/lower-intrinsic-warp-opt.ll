; Check the Kitsune's warp index and lane intrinsics get optimized as expected.
;
; RUN: opt -passes='kit-lower-warp-intrinsics,default<O2>' -S %s \
; RUN:     | FileCheck %s
;

; CHECK-LABEL: @id1
; CHECK-NEXT: %[[X:.+]] = tail call i32 @llvm.kit.gpu.thread.id.x(i32 2)
; CHECK-NEXT: %[[RESULT:.+]] = lshr i32 %[[X]], 5
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @id1() {
  %1 = call i32 @llvm.kit.gpu.warp.id(i32 2, i32 1)
  ret i32 %1
}

; CHECK-LABEL: @id2
; CHECK-NEXT: %[[BSZX:.+]] = tail call i32 @llvm.kit.gpu.block.size.x(i32 2)
; CHECK-NEXT: %[[TIDY:.+]] = tail call i32 @llvm.kit.gpu.thread.id.y(i32 2)
; CHECK-NEXT: %[[Y:.+]] = mul i32 %[[TIDY]], %[[BSZX]]
; CHECK-NEXT: %[[X:.+]] = tail call i32 @llvm.kit.gpu.thread.id.x(i32 2)
; CHECK-NEXT: %[[OFFXY:.+]] = add i32 %[[X]], %[[Y]]
; CHECK-NEXT: %[[RESULT:.+]] = lshr i32 %[[OFFXY]], 5
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @id2() {
  %1 = call i32 @llvm.kit.gpu.warp.id(i32 2, i32 2)
  ret i32 %1
}

; CHECK-LABEL: @id3
; CHECK-NEXT: %[[BSZX:.+]] = tail call i32 @llvm.kit.gpu.block.size.x(i32 2)
; CHECK-NEXT: %[[BSZY:.+]] = tail call i32 @llvm.kit.gpu.block.size.y(i32 2)
; CHECK-NEXT: %[[TIDZ:.+]] = tail call i32 @llvm.kit.gpu.thread.id.z(i32 2)
; CHECK-NEXT: %[[Z:.+]] = mul i32 %[[TIDZ]], %[[BSZY]]
; CHECK-NEXT: %[[TIDY:.+]] = tail call i32 @llvm.kit.gpu.thread.id.y(i32 2)
; CHECK-NEXT: %[[X:.+]] = tail call i32 @llvm.kit.gpu.thread.id.x(i32 2)
; CHECK-NEXT: %[[ADD:.+]] = add i32 %[[TIDY]], %[[Z]]
; CHECK-NEXT: %[[MUL:.+]] = mul i32 %[[ADD]], %[[BSZX]]
; CHECK-NEXT: %[[OFFXYZ:.+]] = add i32 %[[MUL]], %[[X]]
; CHECK-NEXT: %[[RESULT:.+]] = lshr i32 %[[OFFXYZ]], 5
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @id3() {
  %1 = call i32 @llvm.kit.gpu.warp.id(i32 2, i32 3)
  ret i32 %1
}

; CHECK-LABEL: @lane1
; CHECK-NEXT: %[[X:.+]] = tail call i32 @llvm.kit.gpu.thread.id.x(i32 2)
; CHECK-NEXT: %[[RESULT:.+]] = and i32 %[[X]], 31
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @lane1() {
  %1 = call i32 @llvm.kit.gpu.warp.lane(i32 2, i32 1)
  ret i32 %1
}

; CHECK-LABEL: @lane2
; CHECK-NEXT: %[[BSZX:.+]] = tail call i32 @llvm.kit.gpu.block.size.x(i32 2)
; CHECK-NEXT: %[[TIDY:.+]] = tail call i32 @llvm.kit.gpu.thread.id.y(i32 2)
; CHECK-NEXT: %[[Y:.+]] = mul i32 %[[TIDY]], %[[BSZX]]
; CHECK-NEXT: %[[X:.+]] = tail call i32 @llvm.kit.gpu.thread.id.x(i32 2)
; CHECK-NEXT: %[[OFFXY:.+]] = add i32 %[[X]], %[[Y]]
; CHECK-NEXT: %[[RESULT:.+]] = and i32 %[[OFFXY]], 31
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @lane2() {
  %1 = call i32 @llvm.kit.gpu.warp.lane(i32 2, i32 2)
  ret i32 %1
}

; CHECK-LABEL: @lane3
; CHECK-NEXT: %[[BSZX:.+]] = tail call i32 @llvm.kit.gpu.block.size.x(i32 2)
; CHECK-NEXT: %[[BSZY:.+]] = tail call i32 @llvm.kit.gpu.block.size.y(i32 2)
; CHECK-NEXT: %[[TIDZ:.+]] = tail call i32 @llvm.kit.gpu.thread.id.z(i32 2)
; CHECK-NEXT: %[[Z:.+]] = mul i32 %[[TIDZ]], %[[BSZY]]
; CHECK-NEXT: %[[TIDY:.+]] = tail call i32 @llvm.kit.gpu.thread.id.y(i32 2)
; CHECK-NEXT: %[[X:.+]] = tail call i32 @llvm.kit.gpu.thread.id.x(i32 2)
; CHECK-NEXT: %[[ADD:.+]] = add i32 %[[TIDY]], %[[Z]]
; CHECK-NEXT: %[[MUL:.+]] = mul i32 %[[ADD]], %[[BSZX]]
; CHECK-NEXT: %[[OFFXYZ:.+]] = add i32 %[[MUL]], %[[X]]
; CHECK-NEXT: %[[RESULT:.+]] = and i32 %[[OFFXYZ]], 31
; CHECK-NEXT: ret i32 %[[RESULT]]
define i32 @lane3() {
  %1 = call i32 @llvm.kit.gpu.warp.lane(i32 2, i32 3)
  ret i32 %1
}
