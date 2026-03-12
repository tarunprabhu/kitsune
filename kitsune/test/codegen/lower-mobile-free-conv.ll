; Check that the calling convention is preserved when lowering the
; llvm.kit.mobile.free intrinsic.
;
; ------------------------------------------------------------------------------
; RUN: opt -tapir=nolo -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes NOLO %s
;
; NOLO: define {{.+}} @deallocate(ptr addrspace(67) %[[P:.+]])
; NOLO-NEXT: call fastcc void @llvm.kit.mobile.free(ptr addrspace(67) %[[P]])
; NOLO-NEXT: call coldcc void @llvm.kit.mobile.free(ptr addrspace(67) %[[P]])
; NOLO-NEXT: call anyregcc void @llvm.kit.mobile.free(ptr addrspace(67) %[[P]])
; NOLO-NEXT: ret void
;
; ------------------------------------------------------------------------------
; RUN: opt -tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes SERIAL %s
;
; SERIAL: define {{.+}} @deallocate(ptr addrspace(67) %[[P:.+]])
; SERIAL-NEXT: %[[CST1:[0-9]+]] = addrspacecast ptr addrspace(67) %[[P]] to ptr
; SERIAL-NEXT: call fastcc void @free(ptr %[[CST1]])
; SERIAL-NEXT: %[[CST2:[0-9]+]] = addrspacecast ptr addrspace(67) %[[P]] to ptr
; SERIAL-NEXT: call coldcc void @free(ptr %[[CST2]])
; SERIAL-NEXT: %[[CST3:[0-9]+]] = addrspacecast ptr addrspace(67) %[[P]] to ptr
; SERIAL-NEXT: call anyregcc void @free(ptr %[[CST3]])
; SERIAL-NEXT: ret void
;
; ------------------------------------------------------------------------------

target triple = "x86_64-pc-linux-gnu"

define void @deallocate(ptr addrspace(67) %p) {
  call fastcc void @llvm.kit.mobile.free(ptr addrspace(67) %p)
  call coldcc void @llvm.kit.mobile.free(ptr addrspace(67) %p)
  call anyregcc void @llvm.kit.mobile.free(ptr addrspace(67) %p)
  ret void
}
