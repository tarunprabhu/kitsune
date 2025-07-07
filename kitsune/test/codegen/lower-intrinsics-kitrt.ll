; Check that the kitsune intrinsics common to all tapir targets are lowered
; correctly.
;
; ------------------------------------------------------------------------------
; RUN: opt --tapir=nolo -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s --check-prefix NOLO
;
; NOLO-LABEL: @f
; NOLO-NEXT: call void @llvm.kit.enable.verbose(i8 1)
; NOLO-NEXT: ret void
;
; NOLO-LABEL: @g
; NOLO-NEXT: call void @llvm.kit.enable.verbose(i8 0)
; NOLO-NEXT: ret void
;
; ------------------------------------------------------------------------------
; RUN: opt --tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s --check-prefix SERIAL
;
; SERIAL-LABEL: @f
; SERIAL-NEXT: call void @__kitrt_enable_verbose_mode
; SERIAL-NEXT: ret void
;
; SERIAL-LABEL: @g
; SERIAL-NOT: call void @__kitrt_enable_verbose_mode
; SERIAL-NEXT: ret void
;
; ------------------------------------------------------------------------------

target triple = "x86_64-unknown-linux-gnu"

define void @f() {
  call void @llvm.kit.enable.verbose(i8 1)
  ret void
}

define void @g() {
  call void @llvm.kit.enable.verbose(i8 0)
  ret void
}
