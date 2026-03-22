; If a global variable that is expected to contain embedded bitcode was found,
; but the contents of the initializer cannot be parsed into an LLVM module,
; fail with an appropriate error.
;
; RUN: not %kit-mbc -S -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: error: could not parse embedded bitcode

@bc = constant [4 x i8] c"BC\C0\DE", !kit.gv.bit.code !0

!0 = !{i32 2}
