; Check that debug information is preserved when lowering llvm.kit.mobile.alloc
; intrinsics.
;
; ------------------------------------------------------------------------------
; RUN: opt --tapir=nolo -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes NOLO,DEBUG %s
;
; NOLO: define {{.+}} @allocate(i64 %[[N:.+]])
; NOLO-NEXT: #dbg_value
; NOLO-NEXT: %[[PTR:[0-9]+]] = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %[[N]]), !dbg ![[LOC:[0-9]+]]
; NOLO-NEXT: ret ptr addrspace(67) %[[PTR]]
;
; ------------------------------------------------------------------------------
; RUN: opt --tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes SERIAL,DEBUG %s
;
; SERIAL: define {{.+}} @allocate(i64 %[[N:.+]])
; SERIAL-NEXT: #dbg_value
; SERIAL-NEXT: %[[PTR:[0-9]+]] = call noalias ptr @malloc(i64 %[[N]]), !dbg ![[LOC:[0-9]+]]
; SERIAL-NEXT: %[[CST:[0-9]]] = addrspacecast ptr %[[PTR]] to ptr addrspace(67)
; SERIAL-NEXT: ret ptr addrspace(67) %[[CST]]
;
; ------------------------------------------------------------------------------
;
; DEBUG: ![[SCOPE:[0-9]+]] = distinct !DISubprogram(name: "allocate",
; DEBUG: ![[LOC]] = !DILocation(line: 4, column: 10, scope: ![[SCOPE]])
;
; ------------------------------------------------------------------------------

target triple = "x86_64-pc-linux-gnu"

define noalias ptr addrspace(67) @allocate(i64 %n) !dbg !10 {
    #dbg_value(i64 %n, !16, !DIExpression(), !17)
  %1 = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %n), !dbg !18
  ret ptr addrspace(67) %1, !dbg !19
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 20.1.2", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "0d16e345e627737fd088489342372a73")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 20.1.2"}
!10 = distinct !DISubprogram(name: "allocate", scope: !1, file: !1, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !15)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !14}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!14 = !DIBasicType(name: "long long", size: 64, encoding: DW_ATE_signed)
!15 = !{!16}
!16 = !DILocalVariable(name: "n", arg: 1, scope: !10, file: !1, line: 3, type: !14)
!17 = !DILocation(line: 0, scope: !10)
!18 = !DILocation(line: 4, column: 10, scope: !10)
!19 = !DILocation(line: 4, column: 3, scope: !10)
