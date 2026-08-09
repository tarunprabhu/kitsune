; Check that the runtime initialization options are set correctly when multiple
; tapir targets and instrumentation is used.
;
; RUN: opt --tapir=serial -passes=kit-ctors -S %s | FileCheck %s
;
; CHECK: @[[INITOPTS:.+]] = internal constant [8 x i8]
; CHECK-SAME: c"\01\04\00\00\03\00\00\00"
;
; CHECK: call void @__kitrt_initialize(ptr @[[INITOPTS]])
; CHECK: call void @__kitrt_finalize(ptr @[[INITOPTS]])

@e1 = private unnamed_addr constant [3 x i8] c"e1\00"
@e2 = private unnamed_addr constant [3 x i8] c"e2\00"
@evt = private unnamed_addr constant [4 x i8] c"ins\00"

define void @f() {
entry:
  %pctx = call ptr(ptr, i64, i32, ...) @__kitpapi_start(ptr @e1, i64 0, i32 1, ptr @evt)
  call void @__kitpapi_stop(ptr %pctx)
  %tctx = call ptr(ptr, i64) @__kittimer_start(ptr @e2, i64 0)
  %span = call i64 @__kittimer_stop(ptr %tctx)
  ret void
}

declare ptr @__kitpapi_start(ptr, i64, i32, ...)
declare void @__kitpapi_stop(ptr)
declare ptr @__kittimer_start(ptr, i64)
declare void @__kittimer_stop(ptr)

!kit.module = !{!0}

!0 = distinct !{!0, !1}
!1 = !{!"kit.module.tts", !2}
!2 = !{i32 1, i32 1024}
