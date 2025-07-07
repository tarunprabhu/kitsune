; Check that debug information is preserved when lowering llvm.kit.mobile.free
; intrinsics.
;
; ------------------------------------------------------------------------------
; RUN: opt --tapir=none -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes NONE,DEBUG %s
;
; NONE: define {{.+}} @deallocate(ptr addrspace(67) %[[P:[^)]+]])
; NONE-NEXT: #dbg_value
; NONE-NEXT: call void @llvm.kit.mobile.free(ptr addrspace(67) %[[P]]), !dbg ![[LOC:[0-9]+]]
; NONE-NEXT: ret void
;
; ------------------------------------------------------------------------------
; RUN: opt --tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes SERIAL,DEBUG %s
;
; SERIAL: @deallocate(ptr addrspace(67) %[[P:[^)]+]])
; SERIAL-NEXT: #dbg_value
; SERIAL-NEXT: %[[CST:[0-9]]] = addrspacecast ptr addrspace(67) %[[P]] to ptr
; SERIAL-NEXT: call void @free(ptr %[[CST]]), !dbg ![[LOC:[0-9]+]]
; SERIAL-NEXT: ret void
;
; ------------------------------------------------------------------------------
;
; DEBUG: ![[SCOPE:[0-9]+]] = distinct !DISubprogram(name: "deallocate",
; DEBUG: ![[LOC]] = !DILocation(line: 2, column: 2, scope: ![[SCOPE]])
;
; ------------------------------------------------------------------------------

target triple = "x86_64-unknown-linux-gnu"

define void @deallocate(ptr addrspace(67) %p) !dbg !10 {
    #dbg_value(ptr addrspace(67) %p, !15, !DIExpression(), !16)
  call void @llvm.kit.mobile.free(ptr addrspace(67) %p), !dbg !17
  ret void, !dbg !18
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 20.1.2", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "3a87a77e7795175629d53d975e94dd00")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 20.1.2"}
!10 = distinct !DISubprogram(name: "deallocate", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!14 = !{!15}
!15 = !DILocalVariable(name: "p", arg: 1, scope: !10, file: !1, line: 1, type: !13)
!16 = !DILocation(line: 0, scope: !10)
!17 = !DILocation(line: 2, column: 2, scope: !10)
!18 = !DILocation(line: 3, column: 1, scope: !10)
