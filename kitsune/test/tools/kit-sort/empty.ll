; Passing an empty module to kit-sort is ok.
;
; RUN: %kit-sort %s | FileCheck %s
; RUN: echo "" | %kit-sort | FileCheck %s
;
; CHECK: ModuleID =
; CHECK-NEXT: source_filename =
; CHECK-EMPTY:
; CHECK-NOT: {{^.+$}}
