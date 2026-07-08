; Check that the global ctor calls the appropriate functions in Kitsune's
; runtime depending on the command line arguments passed.
;
; RUN: opt --tapir=cuda -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     | FileCheck %s -check-prefix DEFAULT
;
; Currently, even if a max-threads-per-block option is not used, the max is set
; to 1024.
;
; DEFAULT: @[[FB:.+]] = constant [0 x i8] zeroinitializer
; DEFAULT-SAME: section ".nv_fatbin"
; DEFAULT-SAME: !kit.gv ![[MD:[0-9]+]]
;
; DEFAULT: @[[BUNDLE:.+]] = internal constant {{.+}} { i32 1180844977, i32 1, ptr @[[FB]], ptr null }
; DEFAULT-SAME: section ".nvFatBinSegment"
;
; DEFAULT: @[[HANDLE:[.]kitcuda[.].+]] = internal global ptr null
;
; DEFAULT: @llvm.global_ctors = appending global
; DEFAULT-SAME: { i32 65535, ptr @[[CTOR:[.]kitcuda[.]ctor.*]], ptr null }
;
; DEFAULT: define {{.*}} @[[DTOR:[.]kitcuda[.]dtor.*]]{{[ ]*}}(
; DEFAULT: %[[HD:.+]] = load ptr, ptr @[[HANDLE]]
; DEFAULT: call {{.+}} @llvm.kit.gpu.unregister.devcode(i32 2, ptr %[[HD]])
; DEFAULT: call {{.+}} @llvm.kit.runtime.finalize(i32 2)
;
; DEFAULT: define {{.+}} @[[CTOR]]
; DEFAULT: call {{.+}} @llvm.kit.runtime.initialize(i32 2)
; DEFAULT: call {{.+}} @llvm.kit.runtime.set.verbose(i32 2, i8 0)
; DEFAULT-NOT: call {{.+}} @llvm.kit.runtime.set.fixed.tpb(i32 2,
; DEFAULT: call {{.+}} @llvm.kit.runtime.set.max.tpb(i32 2, i32 1024)
; DEFAULT-DAG: call {{.+}} @llvm.kit.runtime.set.kernel.launch.refinement(i32 2, i8 1)
; DEFAULT-DAG: %[[HC:.+]] = call {{.+}}@llvm.kit.gpu.register.devcode(i32 2, ptr @[[BUNDLE]])
; DEFAULT: store ptr %[[HC]], ptr @[[HANDLE]]
; DEFAULT: call void @llvm.kit.gpu.register.devcode.end(i32 2, ptr %[[HC]])
; DEFAULT: call {{.+}}atexit(ptr @[[DTOR]])
; DEFAULT: }
;
; DEFAULT-DAG: ![[MD]] = distinct !{![[MD]], ![[DC:[0-9]+]]}
; DEFAULT-DAG: ![[DC]] = !{!"kit.gv.device.code", i32 2}
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     --tapir-gpu-tpb=77 \
; RUN:     | FileCheck %s -check-prefix TPB
;
; TPB-LABEL: define {{.+}} @.kitcuda.ctor
; TPB: call {{.+}} @llvm.kit.runtime.set.fixed.tpb(i32 2, i32 77)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     --tapir-gpu-max-tpb=29 \
; RUN:     | FileCheck %s -check-prefix MTPB
;
; MTPB-LABEL: define {{.+}} @.kitcuda.ctor
; MTPB: call {{.+}} @llvm.kit.runtime.set.max.tpb(i32 2, i32 29)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda --tapir-verbose \
; RUN:     -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     | FileCheck %s -check-prefix VERBOSE
;
; RUN: opt --tapir=cuda --kitrt-verbose \
; RUN:     -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     | FileCheck %s -check-prefix VERBOSE
;
; VERBOSE-LABEL: define {{.+}} @.kitcuda.ctor
; VERBOSE: call {{.+}} @llvm.kit.runtime.set.verbose(i32 2, i8 1)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     -cuabi-refine-launches=false \
; RUN:     | FileCheck %s -check-prefix NOREFINE
;
; NOREFINE-LABEL: define {{.+}} @.kitcuda.ctor
; NOREFINE: call {{.+}} @llvm.kit.runtime.set.kernel.launch.refinement(i32 2, i8 0)
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
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{!"tapir.loop.perfect.depth", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 1}
