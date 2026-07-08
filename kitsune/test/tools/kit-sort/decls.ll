; If the module only contains functions without bodies, the output must be the
; same as the input
;
; RUN: %kit-sort %s | FileCheck %s
; RUN: cat %s | %kit-sort %s | FileCheck %s
;
; CHECK-DAG: declare void @f1()
; CHECK-DAG: declare void @f2(i32)
; CHECK-DAG: declare ptr @f3(ptr)

declare void @f1()
declare void @f2(i32)
declare ptr @f3(ptr)
