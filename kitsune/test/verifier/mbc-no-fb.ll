; If embedded bitcode is present, a corresponding global containing device code
; must be present as well.
;
; RUN: MBC_2=`%kit-enc --tapir=cuda %s \
; RUN:     | grep "c\"BC" \
; RUN:     | sed 's/.kit.emb//g' \
; RUN:     | sed 's/@.bc/@.bc.cuda/g'`
;
; RUN: MBC_4=`%kit-enc --tapir=hip %s \
; RUN:     | grep "c\"BC" \
; RUN:     | sed 's/.kit.emb//g' \
; RUN:     | sed 's/@.bc/@.bc.hip/g' \
; RUN:     | sed 's/\!0/\!1/g'`
;
; RUN: printf \
; RUN:     "%%s\n%%s\n\n%%s\n%%s\n%%s\n%%s" \
; RUN:     "${MBC_2}" \
; RUN:     "${MBC_4}" \
; RUN:     "!0 = distinct !{!0, !2}" \
; RUN:     "!1 = distinct !{!1, !3}" \
; RUN:     "!2 = !{!\"kit.gv.bit.code\", i32 2}" \
; RUN:     "!3 = !{!\"kit.gv.bit.code\", i32 4}" \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK-DAG: missing device code global for tapir target 'cuda'
; CHECK-DAG: missing device code global for tapir target 'hip'
