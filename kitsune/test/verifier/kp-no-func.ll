; If both a global variable containing embedded bitcode and a global variable
; containing kernel properties exist with the same tapir target, an appropriate
; function must exist in the embedded module.
;
; The command below does the following:
;
;   - Generate a file containing global variables for the embedded bitcode and
;     fat binary. The embedded bitcode is generated from this file
;
;   - Add a global variable that contains properties for the kernel. Note that
;     the value of the kit_kernel_props attribute is "g" whereas the embedded
;     bitcode contains a function named "f"
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | sed -E $'/^[@][.]kit[.]emb[.]fb/a\\\n \
; RUN:         @kp = constant {i64, i64} zeroinitializer #2' \
; RUN:     | sed -E $'/^attributes #1/a\\\n \
; RUN:         attributes #2 = { kit_tt(2) "kit_kernel_props"="g" }' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: global containing properties of non-existent kernel function

define void @f() {
  ret void
}
