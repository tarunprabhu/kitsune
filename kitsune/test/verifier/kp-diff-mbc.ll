; If both a global variable containing embedded bitcode and a global variable
; containing kernel properties exist, but with different tapir targets, it is
; treated as if an embedded bitcode global does not exist for the kernel
; properties global. In other words, it is not an error.
;
; The command below does the following:
;
;   - Generate a file containing global variables for the embedded bitcode and
;     fat binary. The embedded bitcode is generated from this file
;
;   - Add a global variable that contains properties for the kernel. Note that
;     the value of the kit_tt attribute is hip (4) whereas the tapir target
;     for the embedded bitcode module is cuda (2).
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | sed -E $'/^[@][.]kitsune[.]emb[.]fb/a\\\n \
; RUN:         @kp = constant {i64, i64} zeroinitializer #2' \
; RUN:     | sed -E $'/^attributes #1/a\\\n \
; RUN:         attributes #2 = { kit_tt(4) "kit_kernel_props"="f" }' \
; RUN:     | llvm-as -o /dev/null

define void @f() {
  ret void
}
