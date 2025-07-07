; Check that the tail attribute is preserved when lowering the
; llvm.kit.mobile.free intrinsic.
;
; ------------------------------------------------------------------------------
; RUN: opt -tapir=none -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes NONE %s
;
; NONE: define {{.+}} @deallocate(ptr addrspace(67) %[[P:.+]])
; NONE-NEXT: tail call void @llvm.kit.mobile.free(ptr addrspace(67) %[[P]])
; NONE-NEXT: notail call void @llvm.kit.mobile.free(ptr addrspace(67) %[[P]])
; NONE-NEXT: musttail call void @llvm.kit.mobile.free(ptr addrspace(67) %[[P]])
; NONE-NEXT: ret void
;
; ------------------------------------------------------------------------------
; RUN: opt -tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes SERIAL %s
;
; SERIAL: define {{.+}} @deallocate(ptr addrspace(67) %[[P:.+]])
; SERIAL-NEXT: %[[CST1:[0-9]+]] = addrspacecast ptr addrspace(67) %[[P]] to ptr
; SERIAL-NEXT: tail call void @free(ptr %[[CST1]])
; SERIAL-NEXT: %[[CST2:[0-9]+]] = addrspacecast ptr addrspace(67) %[[P]] to ptr
; SERIAL-NEXT: notail call void @free(ptr %[[CST2]])
; SERIAL-NEXT: %[[CST3:[0-9]+]] = addrspacecast ptr addrspace(67) %[[P]] to ptr
; SERIAL-NEXT: tail call void @free(ptr %[[CST3]])
; SERIAL-NEXT: ret void
;
; ------------------------------------------------------------------------------

target triple = "x86_64-unknown-linux-gnu"

define void @deallocate(ptr addrspace(67) %p) {
  tail call void @llvm.kit.mobile.free(ptr addrspace(67) %p)
  notail call void @llvm.kit.mobile.free(ptr addrspace(67) %p)
  musttail call void @llvm.kit.mobile.free(ptr addrspace(67) %p)
  ret void
}
