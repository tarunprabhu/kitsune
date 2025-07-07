; Check that a launch call and a fat binary are present in the host. Check
; that the launch arguments are as expected.
;
; RUN: opt --tapir=cuda -passes='tapir-lowering<O2>' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @[[FB:.+]] = constant {{.+}} #[[FBATTR:[0-9]+]]
; CHECK-DAG: @[[G_KNAME:.+]] = private unnamed_addr constant [{{[0-9]+}} x i8] c"[[KNAME:.+]]\00"
; CHECK-DAG: @[[G_KERNEL_PROPS:.+]] = private unnamed_addr constant {{.+}} zeroinitializer #[[KPATTR:[0-9]+]]
;
; CHECK: define {{.+}} @f(ptr {{.*}}%[[C:.+]], i64 {{.*}}%[[N:.+]])
;
; Local variables for the arguments so everything is passed to the runtime
; launch function.
; CHECK: %[[ARGS:.+]] = alloca [5 x ptr]
; CHECK: %[[LOCAL_TRIP_COUNT:.+]] = alloca i64
; CHECK: %[[LOCAL_START:.+]] = alloca i64
; CHECK: %[[LOCAL_GRAINSIZE:.+]] = alloca i64
; CHECK: %[[LOCAL_C:.+]] = alloca ptr
; CHECK: %[[LOCAL_N:.+]] = alloca i64
;
; The trip count is the first argument of the kernel function.
; CHECK: store i64 %n, ptr %[[LOCAL_TRIP_COUNT]]
; CHECK: store ptr %[[LOCAL_TRIP_COUNT]], ptr %[[ARGS]]
;
; The start index is the second argument.
; CHECK: store i64 0, ptr %[[LOCAL_START]]
; CHECK: %[[ARGS_START:.+]] = getelementptr {{.+}} i8, ptr %[[ARGS]], i64 8
; CHECK: store ptr %[[LOCAL_START]], ptr %[[ARGS_START]]
;
; The grainsize is the third argument. This is usually 1, but don't assume that
; here.
; CHECK: store i64 {{.+}}, ptr %[[LOCAL_GRAINSIZE]]
; CHECK: %[[ARGS_GRAINSIZE:.+]] = getelementptr {{.+}} i8, ptr %[[ARGS]], i64 16
; CHECK: store ptr %[[LOCAL_GRAINSIZE]], ptr %[[ARGS_GRAINSIZE]]
;
; The remaining arguments to the kernel function are whatever else is used in
; the kernel function.
; CHECK: store ptr %[[C]], ptr %[[LOCAL_C]]
; CHECK: %[[ARGS_C:.+]] = getelementptr {{.+}} i8, ptr %[[ARGS]], i64 24
; CHECK: store ptr %[[LOCAL_C]], ptr %[[ARGS_C]]
; CHECK: store i64 %[[N]], ptr %[[LOCAL_N]]
; CHECK: %[[ARGS_N:.+]] = getelementptr {{.+}} i8, ptr %[[ARGS]], i64 32
; CHECK: store ptr %[[LOCAL_N]], ptr %[[ARGS_N]]
;
; The actual launch. The arguments are:
;
;   - tapir target id
;   - fat binary global
;   - kernel name global string literal
;   - trip count
;   - threads per block (zero to indicate that it is unset)
;   - kernel metadata global
;   - thread stream
;
; If the signature of the launch kernel intrinsic changes, this will fail as it
; should.
;
; global, the string literal for the kernel name, the arguments array, the
; trip count,
; CHECK: %[[TS:.+]] = call {{.+}} @llvm.kit.async.launch.kernel(
; CHECK-SAME: i32 2,
; CHECK-SAME: ptr {{.*}}@[[FB]],
; CHECK-SAME: ptr {{.*}}@[[G_KNAME]],
; CHECK-SAME: ptr {{.*}}%[[ARGS]],
; CHECK-SAME: i64 %n,
; CHECK-SAME: i32 0,
; CHECK-SAME: ptr {{.*}}@[[G_KERNEL_PROPS]],
; CHECK-SAME: ptr {{.*}}%{{[A-Za-z0-9._]+}}
; CHECK-SAME: )
;
; By default, we always enter a sync immediately after the launch. A later
; optimization pass may (re)move this if appropriate
; CHECK: call {{.+}} @llvm.kit.sync.stream(i32 2, ptr %[[TS]])
; CHECK: ret void
; CHECK-NEXT: }
;
; CHECK-DAG: #[[KPATTR]] = { kit_tt(2) "kit_kernel_props"="[[KNAME]]" }
; CHECK-DAG: #[[FBATTR]] = { kit_fb kit_tt(2) }

target triple = "x86_64-pc-linux-gnu"

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i64 %n, 0
  br i1 %cmp5, label %forall.detach.preheader, label %forall.sync

forall.detach.preheader:                          ; preds = %entry
  br label %forall.detach

forall.detach:                                    ; preds = %forall.detach.preheader, %forall.inc
  %indvars.iv = phi i64 [ 0, %forall.detach.preheader ], [ %indvars.iv.next, %forall.inc ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:                                      ; preds = %forall.detach
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  store i64 %n, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:                                       ; preds = %forall.body, %forall.detach
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:                                      ; preds = %forall.inc, %entry
  sync within %syncreg, label %forall.end

forall.end:                                       ; preds = %forall.sync
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}
