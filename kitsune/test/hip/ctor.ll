; Check that the global ctor calls the appropriate functions in Kitsune's
; runtime depending on the command line arguments passed.
;
; RUN: opt --tapir=hip -S %s \
; RUN:     -passes='tapir-lowering<O2>,generate-kitsune-ctors' \
; RUN:     --tapir-hip-arch=gfx906 \
; RUN:     --tapir-hip-sramecc=off \
; RUN:     --tapir-hip-xnack=on \
; RUN:     --tapir-hip-features="-sramecc,+xnack" \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     | FileCheck %s -check-prefix DEFAULT
;
; Currently, even if a max-threads-per-block option is not used, the max is set
; to 1024.
;
; DEFAULT: @[[FB:.+]] = constant [0 x i8] zeroinitializer
; DEFAULT-SAME: section ".hip_fatbin"
; DEFAULT-SAME: !kitsune.fb
;
; DEFAULT: @[[BUNDLE:.+]] = internal constant {{.+}} { i32 1212764230, i32 1, ptr @[[FB]], ptr null }
; DEFAULT-SAME: section ".hipFatBinSegment"
;
; DEFAULT: @[[HANDLE:[.]kithip[.].+]] = internal global ptr null
;
; DEFAULT: @llvm.global_ctors = appending global
; DEFAULT-SAME: { i32 65536, ptr @[[CTOR:[.]kithip[.]ctor.*]], ptr null }
;
; FIXME: There is a bug where calling __kithip_destroy raises a segmentation
; fault or some other error which looks like memory corruption bug. As a
; temporary workaround, __kithip_destroy is not called, but it eventually
; should be once the issue is fixed.
;
; DEFAULT: define {{.*}} @[[DTOR:[.]kithip[.]dtor.*]]{{[ ]*}}(
; DEFAULT: call {{.+}} @__hipUnregisterFatBinary
; DEFAULT-NOT: call {{.+}} @llvm.kitrt.finalize(i8 3)
;
; DEFAULT: define {{.+}} @[[CTOR]]
; DEFAULT: call {{.+}} @llvm.kitrt.initialize(i8 3)
; DEFAULT: call {{.+}} @llvm.kitrt.enable.verbose(i8 0)
; DEFAULT: call {{.+}} @llvm.kitrt.hip.enable.xnack(i8 1)
; DEFAULT: call {{.+}} @llvm.kitrt.enable.y.axis.launches(i8 3, i8 0)
; DEFAULT-NOT: call {{.+}} @llvm.kitrt.set.fixed.tpb(i8 3,
; DEFAULT: call {{.+}} @llvm.kitrt.set.max.tpb(i8 3, i32 1024)
; DEFAULT-DAG: %[[HC:.+]] = call {{.+}}__hipRegisterFatBinary(ptr @[[BUNDLE]])
; DEFAULT: store ptr %[[HC]], ptr @[[HANDLE]]
; DEFAULT: call {{.+}}atexit(ptr @[[DTOR]])
; DEFAULT: }
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=hip -S %s \
; RUN:     -passes='tapir-lowering<O2>,generate-kitsune-ctors' \
; RUN:     --tapir-gpu-tpb=77 \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     | FileCheck %s -check-prefix TPB
;
; TPB-LABEL: kithip.ctor{{.*}}
; TPB: call {{.+}} @llvm.kitrt.set.fixed.tpb(i8 3, i32 77)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=hip -S %s \
; RUN:     -passes='tapir-lowering<O2>,generate-kitsune-ctors' \
; RUN:     --tapir-gpu-max-tpb=29 \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     | FileCheck %s -check-prefix MTPB
;
; MTPB-LABEL: kithip.ctor{{.*}}
; MTPB: call {{.+}} @llvm.kitrt.set.max.tpb(i8 3, i32 29)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=hip -S %s \
; RUN:     -passes='tapir-lowering<O2>,generate-kitsune-ctors' \
; RUN:     --tapir-verbose \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     | FileCheck %s -check-prefix VERBOSE
;
; RUN: opt --tapir=hip -S %s \
; RUN:     -passes='tapir-lowering<O2>,generate-kitsune-ctors' \
; RUN:     --kitrt-verbose \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     | FileCheck %s -check-prefix VERBOSE
;
; VERBOSE-LABEL: kithip.ctor{{.*}}
; VERBOSE: call {{.+}} @llvm.kitrt.enable.verbose(i8 1)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=hip -S %s \
; RUN:     -passes='tapir-lowering<O2>,generate-kitsune-ctors' \
; RUN:     --tapir-hip-xnack=off \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     | FileCheck %s -check-prefix NOXNACK
;
; RUN: opt --tapir=hip -S %s \
; RUN:     -passes='tapir-lowering<O2>,generate-kitsune-ctors' \
; RUN:     --tapir-hip-xnack=any \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     | FileCheck %s -check-prefix NOXNACK
;
; NOXNACK-LABEL: kithip.ctor{{.*}}
; NOXNACK: call {{.+}} @llvm.kitrt.hip.enable.xnack(i8 0)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=hip -S %s \
; RUN:     -passes='tapir-lowering<O2>,generate-kitsune-ctors' \
; RUN:     -hipabi-y-launch \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     | FileCheck %s -check-prefix YLAUNCH
;
; YLAUNCH-LABEL: kithip.ctor{{.*}}
; YLAUNCH: call {{.+}} @llvm.kitrt.enable.y.axis.launches(i8 3, i8 1)
;
; ----------------------------------------------------------------------------

; ModuleID = 'clopts.c'
source_filename = "clopts.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

target triple = "x86_64-pc-linux-gnu"

; Function Attrs: nounwind memory(argmem: write) uwtable
define dso_local void @f(ptr nocapture noundef writeonly %c, i32 noundef %n) #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i32 %n, 0
  br i1 %cmp5, label %forall.detach.preheader, label %forall.sync

forall.detach.preheader:                          ; preds = %entry
  %wide.trip.count = zext nneg i32 %n to i64
  br label %forall.detach

forall.detach:                                    ; preds = %forall.detach.preheader, %forall.inc
  %indvars.iv = phi i64 [ 0, %forall.detach.preheader ], [ %indvars.iv.next, %forall.inc ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:                                      ; preds = %forall.detach
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  store i32 %n, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:                                       ; preds = %forall.body, %forall.detach
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:                                      ; preds = %forall.inc, %entry
  sync within %syncreg, label %forall.end

forall.end:                                       ; preds = %forall.sync
  ret void
}

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare token @llvm.syncregion.start() #1

attributes #0 = { nounwind memory(argmem: write) uwtable }
attributes #1 = { mustprogress nounwind willreturn memory(argmem: readwrite) }

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}
