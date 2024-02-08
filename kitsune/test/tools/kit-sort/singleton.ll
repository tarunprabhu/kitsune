; Check that sorting a function containing a single block works as expected.
;
; RUN: %kit-sort %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: ret void
; CHECK-NEXT: }

define void @f() {
  ret void
}
