; Check that the runtime initialization options are set correctly when timing
; is enabled.
;
; RUN: opt --tapir=serial -passes=kit-ctors -S %s | FileCheck %s
;
; CHECK: @[[INITOPTS:.+]] = internal constant [8 x i8]
; CHECK-SAME: c"\01\00\00\00\02\00\00\00"
; CHECK-SAME: section ".kit.rtiopt"
;
; CHECK: call void @__kitrt_initialize(ptr @[[INITOPTS]])
; CHECK: call void @__kitrt_finalize(ptr @[[INITOPTS]])

@e = private unnamed_addr constant [2 x i8] c"e\00"

define void @f() {
entry:
  %e = call ptr(ptr, i64) @__kittimer_start(ptr @e, i64 0)
  call void @__kittimer_stop(ptr %e)
  ret void
}

declare ptr @__kittimer_start(ptr, i64)
declare void @__kittimer_stop(ptr)

!kit.module = !{!0}

!0 = distinct !{!0, !1}
!1 = !{!"kit.module.tts", !2}
!2 = !{i32 1}
