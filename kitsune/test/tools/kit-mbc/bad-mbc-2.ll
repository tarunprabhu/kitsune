; If a global variable that is expected to contain embedded bitcode was found,
; but the contents of the initializer cannot be parsed into an LLVM module,
; fail with an appropriate error.
;
; RUN: cat %s \
; RUN:     | not %kit-mbc -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; RUN: cat %s \
; RUN:     | sed 's/i32 2/i32 4/g' \
; RUN:     | not %kit-mbc -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: error: could not parse embedded bitcode

@bc = constant [2 x i8] c"BC", !kit.gv !0

!0 = distinct !{!0, !1}
!1 = !{!"kit.gv.bit.code", i32 2}
