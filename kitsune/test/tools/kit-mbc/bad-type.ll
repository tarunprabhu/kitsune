; If a global variable that is expected to contain embedded bitcode was found,
; it must be an array of bytes
;
; RUN: not %kit-mbc -S -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: global containing embedded bitcode must be a byte array

@bc = constant [2 x i32] [i32 1, i32 2], !kit.gv !0

!0 = distinct !{!0, !1}
!1 = !{!"kit.gv.bit.code", i32 2}
