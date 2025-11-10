; Currently, we expect all the code embedded bitcode from a given tapir target
; must be combined into a single global. If more than one global with the kit_bc
; attribute is found for a given tapir target, it is an error.
;
; RUN: MBC_4=`%kit-enc --tapir=hip %s \
; RUN:     | grep "c\"BC" \
; RUN:     | sed 's/.kitsune.emb//g' \
; RUN:     | sed 's/@.bc/@.bc.4/g'`
;
; RUN: MBC_H=`%kit-enc --tapir=hip %s \
; RUN:     | grep "c\"BC" \
; RUN:     | sed 's/.kitsune.emb//g' \
; RUN:     | sed 's/@.bc/@.bc.h/g' \
; RUN:     | sed 's/#0/#1/g'`
;
; RUN: printf "%%s\n%%s\n%%s\n%%s\n\n%%s\n%%s\n%%s\n%%s" \
; RUN:         "${MBC_4}" \
; RUN:         "${MBC_H}" \
; RUN:         "@fb.4 = constant [0 x i8] zeroinitializer #2" \
; RUN:         "@fb.h = constant [0 x i8] zeroinitializer #3" \
; RUN:         "attributes #0 = { kit_bc kit_tt(4) }" \
; RUN:         "attributes #1 = { kit_bc kit_tt(4) }" \
; RUN:         "attributes #2 = { kit_fb kit_tt(4) }" \
; RUN:         "attributes #3 = { kit_fb kit_tt(4) }" \
; RUN:     | not llvm-as -o /dev/null
;
; CHECK: too many embedded bitcode globals for tapir target 'hip'
