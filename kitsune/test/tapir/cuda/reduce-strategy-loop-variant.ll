; If the destination of a reduction is loop-variant, only the Direct strategy
; may be used. This cannot be overridden by any command-line option.
;
; RUN: opt -passes='kit-reduce-strategy' -S %s \
; RUN:     | FileCheck %s --check-prefixes=ALL,WSHFL
;
; RUN: opt -passes='kit-reduce-strategy' -kit-gpu-reduce-strategy=direct -S %s \
; RUN:     | FileCheck %s --check-prefixes=ALL,DIRECT
;
; RUN: opt -passes='kit-reduce-strategy' -kit-gpu-reduce-strategy=shmem -S %s \
; RUN:     | FileCheck %s --check-prefixes=ALL,SHMEM
;
; RUN: opt -passes='kit-reduce-strategy' -kit-gpu-reduce-strategy=wshfl -S %s \
; RUN:     | FileCheck %s --check-prefixes=ALL,WSHFL
;
; RUN: opt -passes='kit-reduce-strategy' -kit-gpu-reduce-strategy=wshflshmem -S %s \
; RUN:     | FileCheck %s --check-prefixes=ALL,WSHFLSHMEM

; ALL-LABEL: @f
; ALL-SAME: ptr %[[TOTAL:.[^,]+]]
; ALL-SAME: ptr %[[PARTS:[^,]+]]
; ALL: [[ENTRY:.+]]:
; ALL: [[HEADER:.+]]:
; ALL: %[[I:.+]] = phi i64
; ALL: detach within
; ALL: %[[DEST:.+]] = getelementptr i64, ptr %[[PARTS]], i64 %{{.+}}
; ALL: call {{.+}} @llvm.kit.reduce
; ALL-SAME: i32 2
; ALL-SAME: i32 5
; ALL-SAME: ptr %[[DEST]]
; ALL-SAME: i64 %[[I]]
; ALL-SAME: i32 2
; ALL-NEXT: call {{.+}} @llvm.kit.reduce
; ALL-SAME: i32 2
; ALL-SAME: i32 5
; ALL-SAME: ptr %[[TOTAL]]
; ALL-SAME: i64 %[[I]]
; DIRECT-SAME: i32 2
; SHMEM-SAME: i32 6
; WSHFL-SAME: i32 10
; WSHFLSHMEM-SAME: i32 14
; ALL: reattach within
; ALL: br i1 %{{.+}}, label %{{.+}}, label %[[HEADER]]
define void @f(ptr %total, ptr %parts, i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  %zero.or.one = and i64 %i, 1
  %dest = getelementptr i64, ptr %parts, i64 %zero.or.one
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 2, i32 5, ptr %dest, i64 %i, i32 0, i32 8, i64 0, ptr @sum)
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 2, i32 5, ptr %total, i64 %i, i32 0, i32 8, i64 0, ptr @sum)
  br label %reattach

reattach:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %exit, label %header, !llvm.loop !5

exit:
  ret void
}

declare void @sum(ptr, i64)

!0 = !{!"tapir.loop.target", i32 2}
!1 = !{!"tapir.loop.perfect.level", i32 1}
!2 = !{!"tapir.loop.perfect.depth", i32 1}
!3 = !{!"tapir.loop.reduction"}
!4 = !{!"tapir.loop.prepared"}
!5 = distinct !{!5, !0, !1, !2, !3, !4}
