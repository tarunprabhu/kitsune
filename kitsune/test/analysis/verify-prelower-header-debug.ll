; Only PHI nodes are allowed in the header of a tapir loop. However, debug
; instructions are permitted.
;
; NOTE: The calls to the debug info intrinsics will be automatically upgraded to
; DbgRecord's when this file is parsed. But we keep this test around in case
; that ever changes.
;
; RUN: opt --tapir=nolo -passes=kit-verify-prelower -disable-output %s 2>&1 \
; RUN:     | FileCheck %s --allow-empty
;
; CHECK-NOT: {{^.+$}}

define void @f(i64 %n) !dbg !4 {
entry:
  %syncreg = tail call token @llvm.syncregion.start(), !dbg !11
  br label %header, !dbg !12

header:
  %i = phi i64 [ 0, %entry ], [ %next.i, %latch ]
  call void @llvm.dbg.declare(metadata i64 %i, metadata !9, metadata !DIExpression()), !dbg !11
  detach within %syncreg, label %body, label %latch, !dbg !12

body:
  reattach within %syncreg, label %latch, !dbg !12

latch:
  %next.i = add nuw i64 %i, 1, !dbg !13
  %cmp.i = icmp eq i64 %next.i, %n, !dbg !14
  br i1 %cmp.i, label %exit, label %header, !dbg !17, !llvm.loop !15

exit:
  sync within %syncreg, label %end, !dbg !12

end:
  ret void, !dbg !17
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 21.1.3", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/tmp/test.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "f", scope: !5, file: !5, line: 20, type: !6, scopeLine: 20, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!5 = !DIFile(filename: "test.c", directory: "/tmp")
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9}
!9 = !DILocalVariable(name: "n", arg: 1, scope: !4, file: !1, line: 20, type: !10)
!10 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!11 = !DILocation(line: 0, scope: !4)
!12 = !DILocation(line: 22, column: 3, scope: !4)
!13 = !DILocation(line: 22, column: 32, scope: !4)
!14 = !DILocation(line: 22, column: 27, scope: !4)
!15 = distinct !{!15, !12, !16}
!16 = !{!"tapir.loop.target", i32 1}
!17 = !DILocation(line: 26, column: 1, scope: !4)
