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
; RUN:     | sed 's/.kit.emb//g' \
; RUN:     | sed 's/@.bc/@.bc.2/g'`
;
; RUN: MBC_4=`%kit-enc --tapir=hip %s \
; RUN:     | grep "c\"BC" \
; RUN:     | sed 's/.kit.emb//g' \
; RUN:     | sed 's/@.bc/@.bc.4/g' \
; RUN:     | sed 's/\!0/\!1/g'`
;
; RUN: printf "%%s\n%%s\n%%s\n%%s\n\n%%s\n%%s" \
; RUN:         "${MBC_2}" \
; RUN:         "${MBC_4}" \
; RUN:         "@fb.2 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !0" \
; RUN:         "@fb.4 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !1" \
; RUN:         "!0 = !{i32 2}" \
; RUN:         "!1 = !{i32 4}" \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: 'common' global must have a zero initializer
; CHECK: ptr @0
; CHECK: broken embedded module found
