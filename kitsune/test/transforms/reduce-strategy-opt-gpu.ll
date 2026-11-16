; Check that the -kit-gpu-reduce-strategy option is respected by the
; kit-reduce-strategy pass.
;
; RUN: opt -passes='kit-reduce-strategy' -kit-gpu-reduce-strategy=direct \
; RUN:     -S %s | FileCheck --check-prefixes=DIRECT %s
;
; RUN: opt -passes='kit-reduce-strategy' -kit-gpu-reduce-strategy=shmem \
; RUN:     -S %s | FileCheck --check-prefixes=SHMEM %s
;
; RUN: opt -passes='kit-reduce-strategy' -kit-gpu-reduce-strategy=wshfl \
; RUN:     -S %s | FileCheck --check-prefixes=WSHFL %s
;
; RUN: opt -passes='kit-reduce-strategy' -kit-gpu-reduce-strategy=wshflshmem \
; RUN:     -S %s | FileCheck --check-prefixes=WSHFLSHMEM  %s
;
; DIRECT: call {{.+}} @llvm.kit.reduce
; DIRECT-SAME: i32 2
; DIRECT-SAME: ptr @fadd
; DIRECT: call {{.+}} @llvm.kit.reduce
; DIRECT-SAME: i32 2
; DIRECT-SAME: ptr @iadd
;
; SHMEM: call {{.+}} @llvm.kit.reduce
; SHMEM-SAME: i32 6
; SHMEM-SAME: ptr @fadd
; SHMEM: call {{.+}} @llvm.kit.reduce
; SHMEM-SAME: i32 6
; SHMEM-SAME: ptr @iadd
;
; WSHFL: call {{.+}} @llvm.kit.reduce
; WSHFL-SAME: i32 10
; WSHFL-SAME: ptr @fadd
; WSHFL: call {{.+}} @llvm.kit.reduce
; WSHFL-SAME: i32 10
; WSHFL-SAME: ptr @iadd
;
; WSHFLSHMEM: call {{.+}} @llvm.kit.reduce
; WSHFLSHMEM-SAME: i32 14
; WSHFLSHMEM-SAME: ptr @fadd
; WSHFLSHMEM: call {{.+}} @llvm.kit.reduce
; WSHFLSHMEM-SAME: i32 14
; WSHFLSHMEM-SAME: ptr @iadd

define void @f(ptr %fsum, ptr %isum, i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header1

header1:
  %i1 = phi i64 [ 0, %entry ], [ %inc1, %latch1 ]
  detach within %sr, label %body1, label %latch1

body1:
  %f1 = uitofp i64 %i1 to double
  call void(i32, i32, ptr, double, i32, i32, double, ptr, ...) @llvm.kit.reduce(i32 2, i32 5, ptr %fsum, double %f1, i32 0, i32 8, double 0.0, ptr @fadd)
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 2, i32 5, ptr %isum, i64 %i1, i32 0, i32 8, i64 0, ptr @iadd)
  reattach within %sr, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !5

sync1:
  sync within %sr, label %exit

exit:
  ret void
}

declare void @fadd(ptr, double)
declare void @iadd(ptr, i64)

!0 = !{!"tapir.loop.target", i32 2}
!1 = !{!"tapir.loop.perfect.level", i32 1}
!2 = !{!"tapir.loop.perfect.depth", i32 1}
!3 = !{!"tapir.loop.reduction"}
!4 = !{!"tapir.loop.prepared"}
!5 = distinct !{!5, !0, !1, !2, !3, !4}
