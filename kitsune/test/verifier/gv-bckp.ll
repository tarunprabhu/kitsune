; A global variable cannot have both the kit_bc and kit_kernel_props attributes.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | sed -E \
; RUN:         's/#0 = .+/#0 = { kit_bc kit_tt(2) "kit_kernel_props"="kf" }/g' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: Attributes 'kit_bc' and 'kit_kernel_props' are incompatible

define void @kf() {
  ret void
}
