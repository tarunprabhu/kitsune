; If embedded bitcode is present, a corresponding global containing device code
; must be present as well.
;
; RUN: MBC_2=`%kit-enc --tapir=cuda %s \
; RUN:     | grep "c\"BC" \
; RUN:     | sed 's/.kit.emb//g' \
; RUN:     | sed 's/@.bc/@.bc.2/g'`
; RUN:
; RUN: MBC_4=`%kit-enc --tapir=hip %s \
; RUN:     | grep "c\"BC" \
; RUN:     | sed 's/.kit.emb//g' \
; RUN:     | sed 's/@.bc/@.bc.4/g' \
; RUN:     | sed 's/\!0/\!1/g'`
; RUN:
; RUN: printf "%%s\n%%s\n\n%%s\n%%s" \
; RUN:         "${MBC_2}" \
; RUN:         "${MBC_4}" \
; RUN:         "!0 = !{i32 2}" \
; RUN:         "!1 = !{i32 4}" \
; RUN:     | not llvm-as -o /dev/null
;
; CHECK-COUNT-2: embedded bitcode global without device code global
