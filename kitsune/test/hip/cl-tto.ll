; Check that the command line options make it to the options objects. 
;
; RUN: opt --tapir=hip -passes="tapir-lowering<O2>" -o /dev/null %s \
; RUN:     --tapir-verbose \
; RUN:     --kitrt-verbose \
; RUN:     --tapir-threads-per-block=64 \
; RUN:     --tapir-max-threads-per-block=128 \
; RUN:     --tapir-gpu-prefetch=false \
; RUN:     --tapir-hip-arch=gfx906 \
; RUN:     --tapir-hip-sramecc=off \
; RUN:     --tapir-hip-xnack=on \
; RUN:     --tapir-hip-features="-sramecc,+xnack" \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     --tapir-lld="%S/input/ld.lld" 2>&1 \
; RUN:     | FileCheck %s -check-prefixes ALL,CHECK
;
; RUN: opt --tapir=hip -passes="tapir-lowering<O2>" -o /dev/null %s \
; RUN:     --tapir-verbose --tapir-gpu-prefetch=true 2>&1 \
; RUN:     | FileCheck %s -check-prefixes ALL,PREFETCH
;
; ALL: 'hip' tapir target options
; CHECK:    Runtime verbose: 1
; CHECK:    Optimization level: O2
; CHECK:    GPU fixed threads/block: 64
; CHECK:    GPU max threads/block: 128
; CHECK:    GPU prefetch: 0
; CHECK:    Hip arch: gfx906
; CHECK:    Hip sramecc: off
; CHECK:    Hip xnack: on
; CHECK:    Hip target features: -sramecc,+xnack
; CHECK:    Hip bitcode files: [
; CHECK:      {{.+}}/input/amd.bc
; CHECK:    ]
; PREFETCH: GPU prefetch: 1

; ModuleID = 'clopts.c'
source_filename = "clopts.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind memory(argmem: write) uwtable
define dso_local void @f(ptr nocapture noundef writeonly %c, i32 noundef %n) local_unnamed_addr #0 {
  ret void
}

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare token @llvm.syncregion.start() #1

attributes #0 = { nounwind memory(argmem: write) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 19.1.2 (git@github.com:tarunprabhu/kitsune.git 0ab68f142927b9548ac0bc51a82f9bf5e859b384)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{!9, !10, !11}
!10 = !{!"tapir.loop.spawn.strategy", i32 1}
!11 = !{!"llvm.loop.unroll.disable"}
