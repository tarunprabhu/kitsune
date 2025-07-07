; Check that the command line options make it to the tapir target
;
; RUN: opt --tapir=cuda -passes="tapir-lowering<O2>" -o /dev/null %s \
; RUN:     --tapir-verbose \
; RUN:     --kitrt-verbose \
; RUN:     --tapir-gpu-tpb=64 \
; RUN:     --tapir-gpu-max-tpb=128 \
; RUN:     --tapir-gpu-prefetch=false \
; RUN:     --tapir-cuda-arch=sm_72 \
; RUN:     --tapir-cuda-virt-arch=compute_72 \
; RUN:     --tapir-cuda-features="+ptx72" \
; RUN:     --tapir-cuda-runtime-bc="%S/input/nvptx.bc" 2>&1\
; RUN:     | FileCheck %s -check-prefixes ALL,CHECK
;
; RUN: opt --tapir=cuda -passes="tapir-lowering<O2>" -o /dev/null %s \
; RUN:     --tapir-verbose --tapir-gpu-prefetch=true 2>&1 \
; RUN:     | FileCheck %s -check-prefixes ALL,PREFETCH
;
; ALL: 'cuda' tapir target options
; CHECK:    Runtime verbose: 1
; CHECK:    GPU fixed threads/block: 64
; CHECK:    GPU max threads/block: 128
; CHECK:    GPU prefetch: 0
; CHECK:    Cuda arch: sm_72
; CHECK:    Cuda virtual arch: compute_72
; CHECK:    Cuda target features: +ptx72
; CHECK:    Cuda bitcode file: {{.+}}/input/nvptx.bc
; PREFETCH: GPU prefetch: 1

target triple = "x86_64-unknown-linux-gnu"

define void @f(ptr %c, i32 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i32 %n, 0
  br i1 %cmp5, label %preheader, label %forall.sync

preheader:
  %wide.trip.count = zext nneg i32 %n to i64
  br label %forall.detach

forall.detach:
  %indvars.iv = phi i64 [ 0, %preheader ], [ %indvars.iv.next, %forall.inc ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  store i32 %n, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:
  sync within %syncreg, label %forall.end

forall.end:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}
