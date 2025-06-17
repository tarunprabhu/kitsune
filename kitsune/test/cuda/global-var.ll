; Check that any non-const global variables are handled correctly when
; generating kernel launch calls. They should be memcpy'ed to and from the
; before and after launch calls and must be registered with the runtime in
; the ctor for kitsune's runtime.
;
; RUN: opt --tapir=cuda -passes='tapir-lowering<O2>,generate-kitsune-ctors' \
; RUN:     -S %s \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @[[FB:.+]] = constant {{.+}}, !kitsune.fb
; CHECK-DAG: @[[HOSTVAR:.+]] = external {{.+}} i32
; CHECK-DAG: @[[VARNAME:.+]] = private unnamed_addr constant [5 x i8] c"v137\00"
;
; CHECK: define {{.+}} @f
; CHECK: %[[PTR1:.+]] = tail call {{.+}} @llvm.kitrt.symbol.device.ptr(i8 2, ptr nonnull @[[FB]], ptr nonnull @[[VARNAME]])
; CHECK: call {{.+}} @llvm.kitrt.symbol.memcpy.device(i8 2, ptr %[[PTR1]], ptr nonnull @[[HOSTVAR]], i64 4)
; CHECK: %[[TS:.+]] = call {{.+}} @llvm.kitrt.launch.kernel(i8 2, ptr nonnull @[[FB]],
; CHECK: %[[PTR2:.+]] = call {{.+}} @llvm.kitrt.symbol.device.ptr(i8 2, ptr nonnull @[[FB]], ptr nonnull @[[VARNAME]])
; CHECK: call {{.+}} @llvm.kitrt.symbol.memcpy.host(i8 2, ptr nonnull @[[HOSTVAR]], ptr %[[PTR2]], i64 4)
; CHECK: ret void
; CHECK-NEXT: }
;
; CHECK: define {{.+}} @.kitcuda.ctor{{[^(]*}}
; CHECK: %[[HANDLE:.+]] = call ptr @__cudaRegisterFatBinary
; CHECK: call {{.+}} @__cudaRegisterVar(ptr %[[HANDLE]], ptr @[[HOSTVAR]], ptr @[[VARNAME]]
; CHECK: call {{.+}} @__cudaRegisterFatBinaryEnd

target triple = "x86_64-unknown-linux-gnu"

@v137 = external local_unnamed_addr global i32, align 4

define dso_local void @f(ptr nocapture noundef writeonly %c, i64 noundef %n) #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp4.not = icmp eq i64 %n, 0
  br i1 %cmp4.not, label %forall.sync, label %forall.detach

forall.detach:                                    ; preds = %entry, %forall.inc
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:                                      ; preds = %forall.detach
  %0 = load i32, ptr @v137, align 4
  %arrayidx = getelementptr inbounds nuw i32, ptr %c, i64 %i.05
  store i32 %0, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:                                       ; preds = %forall.body, %forall.detach
  %inc = add nuw i64 %i.05, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:                                      ; preds = %forall.inc, %entry
  sync within %syncreg, label %forall.end

forall.end:                                       ; preds = %forall.sync
  ret void
}

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare token @llvm.syncregion.start() #1

attributes #0 = { mustprogress nounwind memory(read, argmem: write, inaccessiblemem: none) uwtable }
attributes #1 = { mustprogress nounwind willreturn memory(argmem: readwrite) }

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}
