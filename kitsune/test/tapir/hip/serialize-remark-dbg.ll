; Check that the serialize pass reports the loop that was serialized together
; with location information, if it is available.
;
; RUN: opt -passes="kit-serialize" %s -S 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,REMARK
;
; RUN: opt -passes="kit-serialize" %s -S \
; RUN:     -serialize-verbose=1 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,REMARK --allow-empty
;
; REMARK: /tmp/pep.c:18:5: serialized loop
; REMARK-NOT: Loop at depth 2
;
; ------------------------------------------------------------------------------
; RUN: opt -passes="kit-serialize" %s -S \
; RUN:     -serialize-verbose=0 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,QUIET --allow-empty
;
; QUIET-NOT: serialized loop
;
; ------------------------------------------------------------------------------
; RUN: opt -passes="kit-serialize" %s -S \
; RUN:     -serialize-verbose=2 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,VERBOSE --allow-empty
;
; VERBOSE: /tmp/pep.c:18:5: serialized loop
; VERBOSE-NEXT: Loop at depth 2
;
; ------------------------------------------------------------------------------
;
; ALL: %syncreg.i = tail call token @llvm.syncregion.start()
; ALL: %i = phi i64
; ALL: detach within %syncreg.i
; ALL-NOT: tail call token @llvm.syncregion.start()
; ALL: call void @ext1
; ALL: %j = phi i64
; ALL-NOT: detach within %syncreg.j
; ALL: call void @ext2
; ALL-NOT: reattach within %syncreg.j
; ALL-NOT: sync within %syncreg.j
; ALL: reattach within %syncreg.i
; ALL: sync within %syncreg.i

; forall (i ...) {
;     ext1(i);
;     forall (j ...) {
;         exit2(i, j);
;     }
; }
define void @pep(i64 %m, i64 %n) !kit.func !56 !dbg !10 {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start(), !dbg !27
    #dbg_value(i64 %m, !16, !DIExpression(), !27)
    #dbg_value(i64 %n, !17, !DIExpression(), !27)
    #dbg_value(i64 0, !18, !DIExpression(), !28)
  br label %for.i.header, !dbg !30

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
    #dbg_value(i64 %i, !18, !DIExpression(), !28)
  detach within %syncreg.i, label %for.i.body, label %for.i.latch, !dbg !30

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start(), !dbg !31
    #dbg_value(i64 %i, !20, !DIExpression(), !32)
  tail call void @ext1(i64 %i), !dbg !33
    #dbg_value(i64 0, !22, !DIExpression(), !34)
  br label %for.j.header, !dbg !35

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
    #dbg_value(i64 %j, !22, !DIExpression(), !34)
  detach within %syncreg.j, label %for.j.body, label %for.j.latch, !dbg !35

for.j.body:
    #dbg_value(i64 %j, !25, !DIExpression(), !36)
  tail call void @ext2(i64 %i, i64 %j), !dbg !37
  reattach within %syncreg.j, label %for.j.latch, !dbg !37

for.j.latch:
  %inc.j = add i64 %j, 1, !dbg !38
    #dbg_value(i64 %inc.j, !22, !DIExpression(), !34)
  %j.not = icmp eq i64 %inc.j, %n, !dbg !39
  br i1 %j.not, label %for.j.exit, label %for.j.header, !dbg !35, !llvm.loop !40

for.j.exit:
  sync within %syncreg.j, label %for.j.end, !dbg !43

for.j.end:
  reattach within %syncreg.i, label %for.i.latch, !dbg !44

for.i.latch:
  %inc.i = add i64 %i, 1, !dbg !45
    #dbg_value(i64 %inc.i, !18, !DIExpression(), !28)
  %i.not = icmp eq i64 %inc.i, %m, !dbg !29
  br i1 %i.not, label %for.i.exit, label %for.i.header, !dbg !30, !llvm.loop !46

for.i.exit:
  sync within %syncreg.i, label %for.i.end, !dbg !48

for.i.end:
  ret void, !dbg !49
}

declare token @llvm.syncregion.start()

declare !dbg !50 void @ext1(i64)

declare !dbg !53 void @ext2(i64, i64)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 21.1.3", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/tmp/pep.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "bba0de9706a06a896f067baa98c260ee")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 21.1.3"}
!10 = distinct !DISubprogram(name: "pep", scope: !11, file: !11, line: 15, type: !12, scopeLine: 15, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !15)
!11 = !DIFile(filename: "/tmp/pep.c", directory: "", checksumkind: CSK_MD5, checksum: "bba0de9706a06a896f067baa98c260ee")
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14, !14}
!14 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!15 = !{!16, !17, !18, !20, !22, !25}
!16 = !DILocalVariable(name: "m", arg: 1, scope: !10, file: !11, line: 15, type: !14)
!17 = !DILocalVariable(name: "n", arg: 2, scope: !10, file: !11, line: 15, type: !14)
!18 = !DILocalVariable(name: "i", scope: !19, file: !11, line: 16, type: !14)
!19 = distinct !DILexicalBlock(scope: !10, file: !11, line: 16, column: 3)
!20 = !DILocalVariable(name: "i", scope: !21, file: !11, line: 16, type: !14)
!21 = distinct !DILexicalBlock(scope: !19, file: !11, line: 16, column: 3)
!22 = !DILocalVariable(name: "j", scope: !23, file: !11, line: 18, type: !14)
!23 = distinct !DILexicalBlock(scope: !24, file: !11, line: 18, column: 5)
!24 = distinct !DILexicalBlock(scope: !21, file: !11, line: 16, column: 35)
!25 = !DILocalVariable(name: "j", scope: !26, file: !11, line: 18, type: !14)
!26 = distinct !DILexicalBlock(scope: !23, file: !11, line: 18, column: 5)
!27 = !DILocation(line: 0, scope: !10)
!28 = !DILocation(line: 0, scope: !19)
!29 = !DILocation(line: 16, column: 25, scope: !21)
!30 = !DILocation(line: 16, column: 3, scope: !19)
!31 = !DILocation(line: 0, scope: !24)
!32 = !DILocation(line: 0, scope: !21)
!33 = !DILocation(line: 17, column: 5, scope: !24)
!34 = !DILocation(line: 0, scope: !23)
!35 = !DILocation(line: 18, column: 5, scope: !23)
!36 = !DILocation(line: 0, scope: !26)
!37 = !DILocation(line: 19, column: 7, scope: !26)
!38 = !DILocation(line: 18, column: 32, scope: !26)
!39 = !DILocation(line: 18, column: 27, scope: !26)
!40 = distinct !{!40, !35, !41, !42}
!41 = !DILocation(line: 19, column: 16, scope: !23)
!42 = !{!"tapir.loop.target", i32 4}
!43 = !DILocation(line: 18, column: 5, scope: !26)
!44 = !DILocation(line: 19, column: 18, scope: !24)
!45 = !DILocation(line: 16, column: 30, scope: !21)
!46 = distinct !{!46, !30, !47, !42, !54, !55}
!47 = !DILocation(line: 19, column: 18, scope: !19)
!48 = !DILocation(line: 16, column: 3, scope: !21)
!49 = !DILocation(line: 20, column: 1, scope: !10)
!50 = !DISubprogram(name: "ext1", scope: !11, file: !11, line: 5, type: !51, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!51 = !DISubroutineType(types: !52)
!52 = !{null, !14}
!53 = !DISubprogram(name: "ext2", scope: !11, file: !11, line: 6, type: !12, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!54 = !{!"tapir.loop.perfect.depth", i32 1}
!55 = !{!"tapir.loop.perfect.level", i32 1}
!56 = distinct !{!56, !57}
!57 = !{!"kit.func.pre.lower.annotate.pass"}
