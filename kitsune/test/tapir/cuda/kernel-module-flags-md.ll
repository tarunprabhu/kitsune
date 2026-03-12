; Check that the tapir target copies the correct module flags metadata from the
; host module to the device module.
;
; RUN: opt --tapir=cuda -passes='loop-spawning' %s \
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

define void @f1(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr i32, ptr %c, i64 %i
  store i64 %n, ptr %arrayidx, align 4
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

!llvm.module.flags = !{!4, !5, !6, !7, !8, !9}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"Dwarf Version", i32 5}
!9 = !{i32 2, !"Debug Info Version", i32 3}
