; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: invalid value of 'kit_kernel_props' attribute. Kernel name cannot be empty

@0 = constant { i64, i64, i64, i64 } zeroinitializer #0
attributes #0 = { kit_tt(2) "kit_kernel_props"="" }
