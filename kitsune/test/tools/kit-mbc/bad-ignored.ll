; If a global variable contains embedded bitcode which cannot be parsed into an
; LLVM module, but that module is never requested, it is not an error
;
; RUN: %kit-mbc --tapir=hip -S -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: warning: no embedded bitcode modules found

@bc = constant [2 x i8] c"BC" #0

attributes #0 = { kit_bc kit_tt(2) }
