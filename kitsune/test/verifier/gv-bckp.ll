; A global variable cannot have both the bit.code and kernel.properties
; attributes.
;
; ------------------------------------------------------------------------------
;
; The commands below do the following:
;
;   - The kit-enc command creates a module with one pair of embedded bitcode,
;     and embedded device code, globals
;
;   - The grep commands strip the empty lines and metadata definitions
;
;   - The remaining sed commands add definitions for the metadata.
;
; ------------------------------------------------------------------------------
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | grep -vE "[\!][0-9]+ = " \
; RUN:     | grep -vE "^$" \
; RUN:     | sed '2s/\!2$/\!1/g' \
; RUN:     | sed '2s/$/\n!0 = distinct !{!0, !2, !4}/' \
; RUN:     | sed '3s/$/\n!1 = distinct !{!1, !3}/' \
; RUN:     | sed '4s/$/\n!2 = !{!"kit.gv.bit.code", i32 2}/' \
; RUN:     | sed '5s/$/\n!3 = !{!"kit.gv.device.code", i32 2}/' \
; RUN:     | sed '6s/$/\n!4 = !{!"kit.gv.kernel.properties", i32 2, !"imperial"}/' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; ------------------------------------------------------------------------------
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | grep -vE "[\!][0-9]+ = " \
; RUN:     | grep -vE "^$" \
; RUN:     | sed '2s/\!2$/\!1/g' \
; RUN:     | sed '2s/$/\n!0 = distinct !{!0, !2, !4}/' \
; RUN:     | sed '3s/$/\n!1 = distinct !{!1, !3}/' \
; RUN:     | sed '4s/$/\n!2 = !{!"kit.gv.bit.code", i32 4}/' \
; RUN:     | sed '5s/$/\n!3 = !{!"kit.gv.device.code", i32 4}/' \
; RUN:     | sed '6s/$/\n!4 = !{!"kit.gv.kernel.properties", i32 4, !"selwyn"}/' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; ------------------------------------------------------------------------------
;
; CHECK-DAG: attribute 'kit.gv.bit.code': not compatible with 'kit.gv.kernel.properties'
; CHECK-DAG: attribute 'kit.gv.kernel.properties': not compatible with 'kit.gv.bit.code'

define void @kf() {
  ret void
}
