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
; ------------------------------------------------------------------------------
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | grep -vE "[\!][0-9]+ = " \
; RUN:     | grep -vE "^$" \
; RUN:     | sed '2s/\!2$/\!1/' \
; RUN:     | sed '2s/$/\n@kp = constant { i64, i64, i64, i64 } zeroinitializer, !kit.gv !2/' \
; RUN:     | sed '3s/$/\n!0 = distinct !{!0, !3}/' \
; RUN:     | sed '4s/$/\n!1 = distinct !{!1, !4}/' \
; RUN:     | sed '5s/$/\n!2 = distinct !{!2, !5}/' \
; RUN:     | sed '6s/$/\n!3 = !{!"kit.gv.bit.code", i32 2}/' \
; RUN:     | sed '7s/$/\n!4 = !{!"kit.gv.device.code", i32 2}/' \
; RUN:     | sed '8s/$/\n!5 = !{!"kit.gv.kernel.properties", i32 2, !"g"}/' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; ------------------------------------------------------------------------------
;
; CHECK: attribute 'kit.gv.kernel.properties': invalid value at index '1'
; CHECK-SAME: Kernel function does not exist in embedded module

define void @f() {
  ret void
}
