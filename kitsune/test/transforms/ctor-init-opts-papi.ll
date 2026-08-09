; Check that the runtime initialization options are set correctly when PAPI
; instrumentation is to be enabled.
;
; RUN: opt --tapir=openmp -passes=kit-ctors -S %s | FileCheck %s
;
; CHECK: @[[INITOPTS:.+]] = internal constant [8 x i8]
; CHECK-SAME: c"\00\02\00\00\01\00\00\00"
;
; CHECK: call void @__kitrt_initialize(ptr @[[INITOPTS]])
; CHECK: call void @__kitrt_finalize(ptr @[[INITOPTS]])

@e = private unnamed_addr constant [2 x i8] c"e\00"
@evt = private unnamed_addr constant [4 x i8] c"ins\00"

define void @f() {
entry:
  %e = call ptr(ptr, i64, i32, ...) @__kitpapi_start(ptr @e, i64 0, i32 1, ptr @evt)
  call void @__kitpapi_stop(ptr %e)
  ret void
}

declare ptr @__kitpapi_start(ptr, i64, i32, ...)
declare void @__kitpapi_stop(ptr)

!kit.module = !{!0}

!0 = distinct !{!0, !1}
!1 = !{!"kit.module.tts", !2}
!2 = !{i32 512}
