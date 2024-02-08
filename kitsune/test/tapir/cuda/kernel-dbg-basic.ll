; Check that debug information makes it to the kernel module. This checks some
; basic debug info nodes.
;
; RUN: opt --tapir=cuda -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: define ptx_kernel void @__kitcuda_loop_test_c_13_3
; CHECK-SAME: !dbg ![[SUBP2:[0-9]+]]
; CHECK: store i64 %j
; CHECK-SAME: !dbg ![[BODY2:[0-9]+]]
; CHECK: br i1 %cmp.j
; CHECK-SAME: !dbg ![[LOOP2:[0-9]+]], !llvm.loop
;
; CHECK-LABEL: define ptx_kernel void @__kitcuda_loop_test_c_11_3
; CHECK-SAME: !dbg ![[SUBP1:[0-9]+]]
; CHECK: store i64 %i
; CHECK-SAME: !dbg ![[BODY1:[0-9]+]]
; CHECK: br i1 %cmp.i
; CHECK-SAME: !dbg ![[LOOP1:[0-9]+]], !llvm.loop
;
; FIXME: There is a bug in the debug handling that results in two
; DICompileUnit's being created. In fact, one DICompileUnit is created for
; each kernel. It seems like most of the debug information from the parent
; function also gets copied over into the kernel module - this is, obviously,
; immensely wasteful. Nevertheless, we check for that here expecting this to
; fail when we eventually fix the issue.
;
; ![[CU1:[0-9]+]] = distinct !DICompileUnit(lang: DW_LANG_C11, file: ![[FILE:[0-9]+]],
; ![[FILE]] = !DIFile(filename: "test.c", directory: "/tmp")
; ![[CU2:[0-9]+]] = distinct !DICompileUnit(lang: DW_LANG_C11, file: ![[FILE:[0-9]+]],
;
; ![[SUBP2]] = distinct !DISubprogram(name: "xlate", scope: ![[CU2]])
; ![[LOOP2]] = !DILocation(line: 13, column: 3)
; ![[BODY2]] = !DILocation(line: 14, column: 5)
;
; ![[SUBP1]] = distinct !DISubprogram(name: "xlate", scope: ![[CU1]])
; ![[LOOP1]] = !DILocation(line: 11, column: 3)
; ![[BODY1]] = !DILocation(line: 12, column: 5)

define void @xlate(ptr %a, ptr %b, i64 %n) !dbg !10 {
entry:
  %syncreg2 = tail call token @llvm.syncregion.start(), !dbg !28
    #dbg_value(ptr %a, !17, !DIExpression(), !28)
    #dbg_value(ptr %b, !18, !DIExpression(), !28)
    #dbg_value(i64 %n, !19, !DIExpression(), !28)
    #dbg_value(i64 0, !20, !DIExpression(), !29)
  br label %header, !dbg !31

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
    #dbg_value(i64 %i, !20, !DIExpression(), !29)
  detach within %syncreg2, label %body, label %latch, !dbg !31

body:
    #dbg_value(i64 %i, !22, !DIExpression(), !32)
  %arrayidx = getelementptr i64, ptr %a, i64 %i, !dbg !33
  store i64 %i, ptr %arrayidx, align 8, !dbg !34, !tbaa !35
  reattach within %syncreg2, label %latch, !dbg !33

latch:
  %inc = add i64 %i, 1, !dbg !39
    #dbg_value(i64 %inc, !20, !DIExpression(), !29)
  %cmp.i = icmp eq i64 %inc, %n, !dbg !30
  br i1 %cmp.i, label %sync, label %header, !dbg !31, !llvm.loop !40

sync:
  sync within %syncreg2, label %preheader2, !dbg !43

preheader2:
    #dbg_value(i64 0, !24, !DIExpression(), !44)
  br label %header2, !dbg !45

header2:
  %j = phi i64 [ 0, %preheader2 ], [ %j.next, %latch2 ]
    #dbg_value(i64 %j, !24, !DIExpression(), !44)
  detach within %syncreg2, label %body2, label %latch2, !dbg !45

body2:
    #dbg_value(i64 %j, !26, !DIExpression(), !46)
  %arrayidx10 = getelementptr i64, ptr %b, i64 %j, !dbg !47
  store i64 %j, ptr %arrayidx10, align 8, !dbg !48, !tbaa !35
  reattach within %syncreg2, label %latch2, !dbg !47

latch2:
  %j.next = add i64 %j, 1, !dbg !49
    #dbg_value(i64 %j.next, !24, !DIExpression(), !44)
  %cmp.j = icmp eq i64 %j.next, %n, !dbg !50
  br i1 %cmp.j, label %sync2, label %header2, !dbg !45, !llvm.loop !51

sync2:
  sync within %syncreg2, label %exit, !dbg !53

exit:
  ret void, !dbg !54
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 21.1.3", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 21.1.3"}
!10 = distinct !DISubprogram(name: "xlate", scope: !11, file: !11, line: 10, type: !12, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!11 = !DIFile(filename: "test.c", directory: "/tmp")
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14, !14, !15}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!16 = !{!17, !18, !19, !20, !22, !24, !26}
!17 = !DILocalVariable(name: "a", arg: 1, scope: !10, file: !11, line: 10, type: !14)
!18 = !DILocalVariable(name: "b", arg: 2, scope: !10, file: !11, line: 10, type: !14)
!19 = !DILocalVariable(name: "n", arg: 3, scope: !10, file: !11, line: 10, type: !15)
!20 = !DILocalVariable(name: "i", scope: !21, file: !11, line: 11, type: !15)
!21 = distinct !DILexicalBlock(scope: !10, file: !11, line: 11, column: 3)
!22 = !DILocalVariable(name: "i", scope: !23, file: !11, line: 11, type: !15)
!23 = distinct !DILexicalBlock(scope: !21, file: !11, line: 11, column: 3)
!24 = !DILocalVariable(name: "j", scope: !25, file: !11, line: 13, type: !15)
!25 = distinct !DILexicalBlock(scope: !10, file: !11, line: 13, column: 3)
!26 = !DILocalVariable(name: "j", scope: !27, file: !11, line: 13, type: !15)
!27 = distinct !DILexicalBlock(scope: !25, file: !11, line: 13, column: 3)
!28 = !DILocation(line: 0, scope: !10)
!29 = !DILocation(line: 0, scope: !21)
!30 = !DILocation(line: 11, column: 25, scope: !23)
!31 = !DILocation(line: 11, column: 3, scope: !21)
!32 = !DILocation(line: 0, scope: !23)
!33 = !DILocation(line: 12, column: 5, scope: !23)
!34 = !DILocation(line: 12, column: 10, scope: !23)
!35 = !{!36, !36, i64 0}
!36 = !{!"long", !37, i64 0}
!37 = !{!"omnipotent char", !38, i64 0}
!38 = !{!"Simple C/C++ TBAA"}
!39 = !DILocation(line: 11, column: 30, scope: !23)
!40 = distinct !{!40, !31, !41, !42, !55, !56, !57, !58}
!41 = !DILocation(line: 12, column: 12, scope: !21)
!42 = !{!"tapir.loop.lowering.enabled"}
!43 = !DILocation(line: 11, column: 3, scope: !23)
!44 = !DILocation(line: 0, scope: !25)
!45 = !DILocation(line: 13, column: 3, scope: !25)
!46 = !DILocation(line: 0, scope: !27)
!47 = !DILocation(line: 14, column: 5, scope: !27)
!48 = !DILocation(line: 14, column: 10, scope: !27)
!49 = !DILocation(line: 13, column: 30, scope: !27)
!50 = !DILocation(line: 13, column: 25, scope: !27)
!51 = distinct !{!51, !45, !52, !42, !55, !56, !57, !58}
!52 = !DILocation(line: 14, column: 12, scope: !25)
!53 = !DILocation(line: 13, column: 3, scope: !27)
!54 = !DILocation(line: 15, column: 1, scope: !10)
!55 = !{!"tapir.loop.target", i32 2}
!56 = !{!"tapir.loop.spawn.strategy", i32 3}
!57 = !{!"tapir.loop.perfect.depth", i32 1}
!58 = !{!"tapir.loop.perfect.level", i32 1}
