; Currently, we expect exactly one embedded bitcode module, and exactly one
; embedded device code variable for a given tapir target. It is an error if more
; than one global variable has the bit.code, or device.code, attribute.
;
; The commands below do the following:
;
;   - The kit-enc command creates a module with one pair of embedded bitcode,
;     and embedded device code, globals
;
;   - The grep commands strip the empty lines and metadata definitions
;
;   - The first two sed commands create clones of these globals
;
;   - The next three sed commands fix the metadata on these globals so the
;     nodes on each are distinct
;
;   - The remaining sed commands add definitions for the metadata.
;
; ------------------------------------------------------------------------------
; RUN: echo "" \
; RUN:     | %kit-enc --tapir=cuda \
; RUN:     | grep -vE "[\!][0-9]+ = " \
; RUN:     | grep -vE "^$" \
; RUN:     | sed --posix 'N;s/^\(.*\)\n\(.*\)/\1\n\2\n\1\n\2/' \
; RUN:     | sed '1s/kit[.]emb[.]/kit.emb.1./' \
; RUN:     | sed '2s/kit[.]emb[.]/kit.emb.1./' \
; RUN:     | sed '2s/\!2$/\!1/' \
; RUN:     | sed '3s/\!0$/\!2/' \
; RUN:     | sed '4s/\!2$/\!3/' \
; RUN:     | sed '4s/$/\n!0 = distinct !{!0, !4}/' \
; RUN:     | sed '5s/$/\n!1 = distinct !{!1, !5}/' \
; RUN:     | sed '6s/$/\n!2 = distinct !{!2, !4}/' \
; RUN:     | sed '7s/$/\n!3 = distinct !{!3, !5}/' \
; RUN:     | sed '8s/$/\n!4 = !{!"kit.gv.bit.code", i32 2}/' \
; RUN:     | sed '9s/$/\n!5 = !{!"kit.gv.device.code", i32 2}/' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; ------------------------------------------------------------------------------
; RUN: echo "" \
; RUN:     | %kit-enc --tapir=hip \
; RUN:     | grep -vE "[\!][0-9]+ = " \
; RUN:     | grep -vE "^$" \
; RUN:     | sed --posix 'N;s/^\(.*\)\n\(.*\)/\1\n\2\n\1\n\2/' \
; RUN:     | sed '1s/kit[.]emb[.]/kit.emb.1./' \
; RUN:     | sed '2s/kit[.]emb[.]/kit.emb.1./' \
; RUN:     | sed '2s/\!2$/\!1/' \
; RUN:     | sed '3s/\!0$/\!2/' \
; RUN:     | sed '4s/\!2$/\!3/' \
; RUN:     | sed '4s/$/\n!0 = distinct !{!0, !4}/' \
; RUN:     | sed '5s/$/\n!1 = distinct !{!1, !5}/' \
; RUN:     | sed '6s/$/\n!2 = distinct !{!2, !4}/' \
; RUN:     | sed '7s/$/\n!3 = distinct !{!3, !5}/' \
; RUN:     | sed '8s/$/\n!4 = !{!"kit.gv.bit.code", i32 2}/' \
; RUN:     | sed '9s/$/\n!5 = !{!"kit.gv.device.code", i32 2}/' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; ------------------------------------------------------------------------------
; CHECK-DAG: too many embedded bitcode globals for tapir target '[[TT:.+]]'
; CHECK-DAG: too many embedded device code globals for tapir target '[[TT]]'
