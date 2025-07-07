; Check that the tail call attribute is preserved when lowering
; llvm.kit.mobile.alloc intrinsics.
;
; ------------------------------------------------------------------------------
; RUN: opt --tapir=none -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefix=NONE %s
;
; NONE: define {{.+}} @allocate(i64 %[[N:.+]])
; NONE-NEXT: %[[PTR1:[0-9]+]] = tail call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %[[N]])
; NONE-NEXT: %[[PTR2:[0-9]+]] = notail call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %[[N]])
; NONE-NEXT: %[[PTR3:[0-9]+]] = musttail call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %[[N]])
; NONE-NEXT: ret ptr addrspace(67) %[[PTR3]]
;
; ------------------------------------------------------------------------------
; musttail calls are relaxed to simple tail calls. Since the result of the
; lowered intrinsic must be casted, the tail call cannot be guaranteed. This is
; LLVM's conservative appraisal over which we have no control.
;
; RUN: opt --tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefix=SERIAL %s
;
; SERIAL: define {{.+}} @allocate(i64 %[[N:.+]])
; SERIAL-NEXT: %[[PTR1:[0-9]+]] = tail call noalias ptr @malloc(i64 %[[N]])
; SERIAL-NEXT: %[[CST1:[0-9]]] = addrspacecast ptr %[[PTR1]] to ptr addrspace(67)
; SERIAL-NEXT: %[[PTR2:[0-9]+]] = notail call noalias ptr @malloc(i64 %[[N]])
; SERIAL-NEXT: %[[CST2:[0-9]]] = addrspacecast ptr %[[PTR2]] to ptr addrspace(67)
; SERIAL-NEXT: %[[PTR3:[0-9]+]] = tail call noalias ptr @malloc(i64 %[[N]])
; SERIAL-NEXT: %[[CST3:[0-9]]] = addrspacecast ptr %[[PTR3]] to ptr addrspace(67)
; SERIAL-NEXT: ret ptr addrspace(67) %[[CST3]]
;
; ------------------------------------------------------------------------------

target triple = "x86_64-unknown-linux-gnu"

define noalias ptr addrspace(67) @allocate(i64 %n) {
  %1 = tail call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %n)
  %2 = notail call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %n)
  %3 = musttail call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %n)
  ret ptr addrspace(67) %3
}
