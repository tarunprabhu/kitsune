; If a global variable that is expected to contain embedded bitcode was found,
; it must have an initializer.
;
; RUN: not %kit-mbc -S -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: initializer missing in global containing embedded bitcode

@bc = external global [2 x i8], !kit.gv.bit.code !0

!0 = !{i32 2}
