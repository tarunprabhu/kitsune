; Check that the verifier fails if at least one embedded bitcode module contains
; an error. In this case, the embedded for the cuda tapir target contains an
; invalid global variable - one with common linkage and a non-zero initialized.
; The embedded bitcode for the hip tapir target is valid.
;
; NOTE: This is one of the ways in which we can construct an invalid module
; without the verifier failing. If this is ever fixed in LLVM, we may have to
; find a different way of doing this.
;
; RUN: MBC_2=`echo "@0 = common global i32 11" \
; RUN:     | %kit-enc --tapir=cuda \
; RUN:     | grep "c\"BC" \
; RUN:     | sed 's/.kit.emb//g' \
; RUN:     | sed 's/@.bc/@.bc.cuda/g'`
;
; RUN: MBC_4=`%kit-enc --tapir=hip %s \
; RUN:     | grep "c\"BC" \
; RUN:     | sed 's/.kit.emb//g' \
; RUN:     | sed 's/@.bc/@.bc.hip/g' \
; RUN:     | sed 's/\!0/\!2/g'`
;
; RUN: printf \
; RUN:     "%%s\n%%s\n%%s\n%%s\n\n%%s\n%%s\n%%s\n%%s\n%%s\n%%s\n%%s\n%%s" \
; RUN:     "${MBC_2}" \
; RUN:     "${MBC_4}" \
; RUN:     "@fb.2 = constant [0 x i8] zeroinitializer, !kit.gv !4" \
; RUN:     "@fb.4 = constant [0 x i8] zeroinitializer, !kit.gv !6" \
; RUN:     "!0 = distinct !{!0, !1}" \
; RUN:     "!1 = !{!\"kit.gv.bit.code\", i32 2}" \
; RUN:     "!2 = distinct !{!2, !3}" \
; RUN:     "!3 = !{!\"kit.gv.bit.code\", i32 4}" \
; RUN:     "!4 = distinct !{!4, !5}" \
; RUN:     "!5 = !{!\"kit.gv.device.code\", i32 2}" \
; RUN:     "!6 = distinct !{!6, !7}" \
; RUN:     "!7 = !{!\"kit.gv.device.code\", i32 4}" \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: 'common' global must have a zero initializer
; CHECK: ptr @0
; CHECK: embedded module: broken module found
