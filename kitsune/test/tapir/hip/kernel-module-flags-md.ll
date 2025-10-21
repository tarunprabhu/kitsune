; Check that the tapir target copies the correct module flags metadata from the
; host module to the device module.
;
; RUN: opt %s --tapir=hip -passes='tapir-lowering<O2>' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: !llvm.module.flags = !{![[WCHAR:[0-9]+]], ![[PIC:[0-9]+]], ![[PIE:[0-9]+]], ![[DWARF_VERSION:[0-9]+]], ![[DEBUG_INFO_VERSION:[0-9]+]]
;
; CHECK-NOT: !{i32 7, "!uwtable", i32 2}
; CHECK-DAG: ![[WCHAR]] = !{i32 1, !"wchar_size", i32 4}
; CHECK-DAG: ![[PIC]] = !{i32 8, !"PIC Level", i32 2}
; CHECK-DAG: ![[PIE]] = !{i32 7, !"PIE Level", i32 2}
; CHECK-DAG: ![[DWARF_VERSION]] = !{i32 7, !"Dwarf Version", i32 5}
; CHECK-DAG: ![[DEBUG_INFO_VERSION]] = !{i32 2, !"Debug Info Version", i32 3}

target triple = "x86_64-pc-linux-gnu"

define void @f1(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i64 %n, 0
  br i1 %cmp5, label %preheader, label %forall.sync

preheader:
  br label %forall.detach

forall.detach:
  %indvars.iv = phi i64 [ 0, %preheader ], [ %indvars.iv.next, %forall.inc ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  store i64 %n, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:
  sync within %syncreg, label %forall.end

forall.end:
  ret void
}

!llvm.module.flags = !{!3, !4, !5, !6, !7, !8}

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 2}
!2 = !{!"llvm.loop.unroll.disable"}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 8, !"PIC Level", i32 2}
!5 = !{i32 7, !"PIE Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 2}
!7 = !{i32 7, !"Dwarf Version", i32 5}
!8 = !{i32 2, !"Debug Info Version", i32 3}
