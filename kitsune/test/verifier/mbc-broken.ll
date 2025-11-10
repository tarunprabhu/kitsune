; Check that the verifier fails if at least one embedded bitcode module
; contains an error. In this case, the bitcode for the cuda tapir target
; contains a global variable with an invalid initializer.
;
; NOTE: This is one of the ways in which we can construct an invalid module
; without the verifier failing. If this is ever fixed in LLVM, we may have to
; find a different way of doing this.
;
; RUN: MBC_2=`echo "@0 = common global i32 11" \
; RUN:     | %kit-enc --tapir=cuda \
; RUN:     | grep "c\"BC" \
; RUN:     | sed 's/.kitsune.emb//g' \
; RUN:     | sed 's/@.bc/@.bc.2/g'`
;
; RUN: MBC_4=`%kit-enc --tapir=hip %s \
; RUN:     | grep "c\"BC" \
; RUN:     | sed 's/.kitsune.emb//g' \
; RUN:     | sed 's/@.bc/@.bc.4/g' \
; RUN:     | sed 's/#0/#1/g'`
;
; RUN: printf "%%s\n%%s\n%%s\n%%s\n\n%%s\n%%s\n%%s\n%%s" \
; RUN:         "${MBC_2}" \
; RUN:         "${MBC_4}" \
; RUN:         "@fb.2 = constant [0 x i8] zeroinitializer #2" \
; RUN:         "@fb.4 = constant [0 x i8] zeroinitializer #3" \
; RUN:         "attributes #0 = { kit_bc kit_tt(2) }" \
; RUN:         "attributes #1 = { kit_bc kit_tt(4) }" \
; RUN:         "attributes #2 = { kit_fb kit_tt(2) }" \
; RUN:         "attributes #3 = { kit_fb kit_tt(4) }" \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: 'common' global must have a zero initializer
; CHECK: ptr @0
; CHECK: broken embedded module found
