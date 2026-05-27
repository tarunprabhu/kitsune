; Check that the global ctor calls the appropriate functions in Kitsune's
; runtime depending on the command line arguments passed.
;
; RUN: opt --tapir=hip -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     | FileCheck %s -check-prefix DEFAULT
;
; Currently, even if a max-threads-per-block option is not used, the max is set
; to 1024.
;
; DEFAULT: @[[FB:.+]] = constant [0 x i8] zeroinitializer
; DEFAULT-SAME: section ".hip_fatbin"
; DEFAULT-SAME: !kit.gv ![[MD:[0-9]+]]
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
; DEFAULT-NOT: call {{.+}} @llvm.kit.runtime.finalize(i32 4)
;
; DEFAULT: define {{.+}} @[[CTOR]]
; DEFAULT: call {{.+}} @llvm.kit.runtime.initialize(i32 4)
; DEFAULT: call {{.+}} @llvm.kit.runtime.set.verbose(i8 0)
; DEFAULT: call {{.+}} @llvm.kit.runtime.set.xnack(i8 1)
; DEFAULT: call {{.+}} @llvm.kit.runtime.set.y.axis.kernel.launch(i32 4, i8 0)
; DEFAULT-NOT: call {{.+}} @llvm.kit.runtime.set.fixed.tpb(i32 4,
; DEFAULT: call {{.+}} @llvm.kit.runtime.set.max.tpb(i32 4, i32 1024)
; DEFAULT-DAG: %[[HC:.+]] = call {{.+}}__hipRegisterFatBinary(ptr @[[BUNDLE]])
; DEFAULT: store ptr %[[HC]], ptr @[[HANDLE]]
; DEFAULT: call {{.+}}atexit(ptr @[[DTOR]])
; DEFAULT: }
;
; DEFAULT-DAG: ![[MD]] = distinct !{![[MD]], ![[DC:[0-9]+]]}
; DEFAULT-DAG: ![[DC]] = !{!"kit.gv.device.code", i32 4}
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=hip -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     --tapir-gpu-tpb=77 \
; RUN:     | FileCheck %s -check-prefix TPB
;
; TPB-LABEL: kithip.ctor{{.*}}
; TPB: call {{.+}} @llvm.kit.runtime.set.fixed.tpb(i32 4, i32 77)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=hip -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     --tapir-gpu-max-tpb=29 \
; RUN:     | FileCheck %s -check-prefix MTPB
;
; MTPB-LABEL: kithip.ctor{{.*}}
; MTPB: call {{.+}} @llvm.kit.runtime.set.max.tpb(i32 4, i32 29)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=hip -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     --tapir-verbose \
; RUN:     | FileCheck %s -check-prefix VERBOSE
;
; RUN: opt --tapir=hip -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     --kitrt-verbose \
; RUN:     | FileCheck %s -check-prefix VERBOSE
;
; VERBOSE-LABEL: kithip.ctor{{.*}}
; VERBOSE: call {{.+}} @llvm.kit.runtime.set.verbose(i8 1)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=hip -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     --tapir-hip-xnack=off \
; RUN:     | FileCheck %s -check-prefix NOXNACK
;
; RUN: opt --tapir=hip -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     --tapir-hip-xnack=any \
; RUN:     | FileCheck %s -check-prefix NOXNACK
;
; NOXNACK-LABEL: kithip.ctor{{.*}}
; NOXNACK: call {{.+}} @llvm.kit.runtime.set.xnack(i8 0)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=hip -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     -hipabi-y-launch \
; RUN:     | FileCheck %s -check-prefix YLAUNCH
;
; YLAUNCH-LABEL: kithip.ctor{{.*}}
; YLAUNCH: call {{.+}} @llvm.kit.runtime.set.y.axis.kernel.launch(i32 4, i8 1)
;
; ----------------------------------------------------------------------------

target triple = "x86_64-pc-linux-gnu"

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  reattach within %syncreg, label %latch

latch:
  %i.next = add i64 %i, 1
  %cmp.i = icmp eq i64 %i.next, %n
  br i1 %cmp.i, label %sync, label %header, !llvm.loop !0

sync:
  sync within %syncreg, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{!"tapir.loop.perfect.depth", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 1}
