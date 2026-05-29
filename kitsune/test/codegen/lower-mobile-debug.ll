; Check that debug information is preserved when lowering Kitsune's mobile
; intrinsics. The handling is the same for all tapir targets, so checking this
; with the 'serial' tapir target is sufficient.
;
; RUN: opt --tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @allocate
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK: call ptr @malloc(i64 %[[N]]), !dbg ![[DBG_ALLOC:[0-9]+]]
define void @allocate(i64 %n) !dbg !9 {
  %1 = tail call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1, i64 %n), !dbg !15
  ret void, !dbg !16
}

; CHECK-LABEL: @deallocate
; CHECK-SAME: ptr addrspace(67) %[[P:[^)]+]]
; CHECK: call void @free(ptr %{{.+}}), !dbg ![[DBG_FREE:[0-9]+]]
define void @deallocate(ptr addrspace(67) %ptr) !dbg !17 {
  tail call void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) %ptr), !dbg !19
  ret void, !dbg !20
}

; CHECK-LABEL: @init
; CHECK-SAME: ptr addrspace(67) %[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK: call void @__kitrt_mobile_init_i32(ptr {{[^,]+}}, i64 %[[N]], i32 1)
; CHECK-SAME: !dbg ![[DBG_INIT:[0-9]+]]
define void @init(ptr addrspace(67) %buf, i64 %n) !dbg !21 {
  tail call void (i32, ptr addrspace(67), i64, i32, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) %buf, i64 %n, i32 1), !dbg !23
  ret void, !dbg !24
}

; CHECK-DAG: ![[F_ALLOC:[0-9]+]] = distinct !DISubprogram(name: "allocate"
; CHECK-DAG: ![[DBG_ALLOC]] = !DILocation(line: 4, column: 10, scope: ![[F_ALLOC]])

; CHECK-DAG: ![[F_FREE:[0-9]+]] = distinct !DISubprogram(name: "deallocate"
; CHECK-DAG: ![[DBG_FREE]] = !DILocation(line: 8, column: 3, scope: ![[F_FREE]])

; CHECK-DAG: ![[F_INIT:[0-9]+]] = distinct !DISubprogram(name: "init"
; CHECK-DAG: ![[DBG_INIT]] = !DILocation(line: 12, column: 3, scope: ![[F_INIT]])

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 21.1.3", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/tmp/test.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"clang version 21.1.3"}
!9 = distinct !DISubprogram(name: "allocate", scope: !10, file: !10, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DIFile(filename: "/tmp/test.c", directory: "")
!11 = !DISubroutineType(types: !12)
!12 = !{}
!15 = !DILocation(line: 4, column: 10, scope: !9)
!16 = !DILocation(line: 4, column: 3, scope: !9)
!17 = distinct !DISubprogram(name: "deallocate", scope: !10, file: !10, line: 7, type: !11, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!19 = !DILocation(line: 8, column: 3, scope: !17)
!20 = !DILocation(line: 9, column: 1, scope: !17)
!21 = distinct !DISubprogram(name: "init", scope: !10, file: !10, line: 11, type: !11, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!23 = !DILocation(line: 12, column: 3, scope: !21)
!24 = !DILocation(line: 13, column: 1, scope: !21)
