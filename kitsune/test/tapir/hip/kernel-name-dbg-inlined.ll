; Check that when debug info is available and functions containing forall loops
; are inlined, the names of the kernel functions are derived from the inlined
; location.
;
; The LLVM IR below is obtained from the following C code. Line numbers have
; been included for reference.
;
;       1| #include <kitsune.h>
;       2|
;       3| extern "C" void vecadd(int *c, int *a, int *b, long n) {
;       4|   forall (long i = 0; i < n; i++) {
;       5|     c[i] = a[i] + b[i];
;       6|   }
;       7| }
;       8|
;       9| extern "C" void f(int * c, int *a, long n) {
;      10|   vecadd(c, a, a, n);
;      11| }
;      12|
;      13| extern "C" void g(int * c, int *a, int *b, long n) {
;      14|   vecadd(c, a, b, n);
;      15| }
;
; RUN: opt --tapir=hip -passes='module-inline,loop-spawning' -S %s \
; RUN:     | FileCheck %s
;
; CHECK: @[[KERN_VECADD:.+]] = private {{.+}} c"__kithip_loop_test_cpp_4_3_{{[0-9]}}\00"
; CHECK: @[[KERN_F:.+]] = private {{.+}} c"__kithip_loop_test_cpp_10_3_{{[0-9]}}\00"
; CHECK: @[[KERN_G:.+]] = private {{.+}} c"__kithip_loop_test_cpp_14_3_{{[0-9]}}\00"
;
; CHECK: define void @vecadd
; CHECK: call {{.+}} @llvm.kit.async.gpu.kernel.launch
; CHECK-SAME: ptr @[[KERN_VECADD]]
;
; CHECK: define void @f
; CHECK: call {{.+}} @llvm.kit.async.gpu.kernel.launch
; CHECK-SAME: ptr @[[KERN_F]]
;
; CHECK: define void @g
; CHECK: call {{.+}} @llvm.kit.async.gpu.kernel.launch
; CHECK-SAME: ptr @[[KERN_G]]

define void @vecadd(ptr %c, ptr %a, ptr %b, i64 %n) !dbg !259 {
entry:
  %syncreg = tail call token @llvm.syncregion.start(), !dbg !273
    #dbg_value(ptr %c, !265, !DIExpression(), !273)
    #dbg_value(ptr %a, !266, !DIExpression(), !273)
    #dbg_value(ptr %b, !267, !DIExpression(), !273)
    #dbg_value(i64 %n, !268, !DIExpression(), !273)
    #dbg_value(i64 0, !269, !DIExpression(), !274)
  br label %header, !dbg !276

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
    #dbg_value(i64 %i, !269, !DIExpression(), !274)
  detach within %syncreg, label %body, label %latch, !dbg !276

body:
    #dbg_value(i64 %i, !271, !DIExpression(), !277)
  %arrayidx = getelementptr i32, ptr %a, i64 %i, !dbg !278
  %0 = load i32, ptr %arrayidx, align 4, !dbg !278, !tbaa !280
  %arrayidx2 = getelementptr i32, ptr %b, i64 %i, !dbg !284
  %1 = load i32, ptr %arrayidx2, align 4, !dbg !284, !tbaa !280
  %add = add i32 %1, %0, !dbg !285
  %arrayidx3 = getelementptr i32, ptr %c, i64 %i, !dbg !286
  store i32 %add, ptr %arrayidx3, align 4, !dbg !287, !tbaa !280
  reattach within %syncreg, label %latch, !dbg !288

latch:
  %i.next = add i64 %i, 1, !dbg !289
    #dbg_value(i64 %i.next, !269, !DIExpression(), !274)
  %cmp.i = icmp eq i64 %i.next, %n, !dbg !275
  br i1 %cmp.i, label %sync, label %header, !dbg !276, !llvm.loop !290

sync:
  sync within %syncreg, label %exit, !dbg !293

exit:
  ret void, !dbg !294
}

define void @f(ptr %c, ptr %a, i64 %n) !dbg !295 {
entry:
    #dbg_value(ptr %c, !299, !DIExpression(), !302)
    #dbg_value(ptr %a, !300, !DIExpression(), !302)
    #dbg_value(i64 %n, !301, !DIExpression(), !302)
  call void @vecadd(ptr %c, ptr %a, ptr %a, i64 %n), !dbg !303
  ret void, !dbg !304
}

define void @g(ptr %c, ptr %a, ptr %b, i64 %n) !dbg !305 {
entry:
    #dbg_value(ptr %c, !307, !DIExpression(), !311)
    #dbg_value(ptr %a, !308, !DIExpression(), !311)
    #dbg_value(ptr %b, !309, !DIExpression(), !311)
    #dbg_value(i64 %n, !310, !DIExpression(), !311)
  call void @vecadd(ptr %c, ptr %a, ptr %b, i64 %n), !dbg !312
  ret void, !dbg !313
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!251, !252, !253, !254, !255, !256, !257}
!llvm.ident = !{!258}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 21.1.3 (git@github.com:tarunprabhu/kitsune.git c21cc3bd73b905422e733554fab8dcc1781c85dd)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, imports: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/tmp/test.cpp", directory: "/home/tarun/workspace/kitsune/build", checksumkind: CSK_MD5, checksum: "8aac604d479ae91586e56a9af3432d5d")
!2 = !{!3, !11, !15, !22, !26, !34, !39, !41, !49, !53, !57, !67, !69, !73, !77, !81, !86, !90, !94, !98, !102, !110, !114, !118, !120, !124, !128, !133, !139, !143, !147, !149, !157, !161, !169, !171, !175, !179, !183, !187, !192, !197, !202, !203, !204, !205, !207, !208, !209, !210, !211, !212, !213, !215, !216, !217, !218, !219, !220, !221, !222, !227, !228, !229, !230, !231, !232, !233, !234, !235, !236, !237, !238, !239, !240, !241, !242, !243, !244, !245, !246, !247, !248, !249, !250}
!3 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !5, file: !10, line: 58)
!4 = !DINamespace(name: "std", scope: null)
!5 = !DISubprogram(name: "abs", scope: !6, file: !6, line: 980, type: !7, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!6 = !DIFile(filename: "/usr/include/stdlib.h", directory: "", checksumkind: CSK_MD5, checksum: "2ebb4e08912aad41774217f29ad02c9e")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIFile(filename: "/usr/lib64/gcc/x86_64-pc-linux-gnu/15.2.1/../../../../include/c++/15.2.1/bits/std_abs.h", directory: "", checksumkind: CSK_MD5, checksum: "e447352e9df05640e24a5f9f85d288ce")
!11 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !12, file: !14, line: 137)
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "div_t", file: !6, line: 63, baseType: !13)
!13 = !DICompositeType(tag: DW_TAG_structure_type, file: !6, line: 59, size: 64, flags: DIFlagFwdDecl, identifier: "_ZTS5div_t")
!14 = !DIFile(filename: "/usr/lib64/gcc/x86_64-pc-linux-gnu/15.2.1/../../../../include/c++/15.2.1/cstdlib", directory: "", checksumkind: CSK_MD5, checksum: "745c77d592b579358a91081122d152be")
!15 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !16, file: !14, line: 138)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "ldiv_t", file: !6, line: 71, baseType: !17)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !6, line: 67, size: 128, flags: DIFlagTypePassByValue, elements: !18, identifier: "_ZTS6ldiv_t")
!18 = !{!19, !21}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !17, file: !6, line: 69, baseType: !20, size: 64)
!20 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !17, file: !6, line: 70, baseType: !20, size: 64, offset: 64)
!22 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !23, file: !14, line: 140)
!23 = !DISubprogram(name: "abort", scope: !6, file: !6, line: 730, type: !24, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagOptimized)
!24 = !DISubroutineType(types: !25)
!25 = !{null}
!26 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !27, file: !14, line: 142)
!27 = !DISubprogram(name: "aligned_alloc", scope: !6, file: !6, line: 724, type: !28, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!28 = !DISubroutineType(types: !29)
!29 = !{!30, !31, !31}
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!31 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !32, line: 18, baseType: !33)
!32 = !DIFile(filename: "lib/clang/21/include/__stddef_size_t.h", directory: "/home/tarun/workspace/kitsune/build", checksumkind: CSK_MD5, checksum: "2c44e821a2b1951cde2eb0fb2e656867")
!33 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!34 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !35, file: !14, line: 144)
!35 = !DISubprogram(name: "atexit", scope: !6, file: !6, line: 734, type: !36, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!36 = !DISubroutineType(types: !37)
!37 = !{!9, !38}
!38 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !24, size: 64)
!39 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !40, file: !14, line: 147)
!40 = !DISubprogram(name: "at_quick_exit", scope: !6, file: !6, line: 739, type: !36, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!41 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !42, file: !14, line: 150)
!42 = !DISubprogram(name: "atof", scope: !6, file: !6, line: 102, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!43 = !DISubroutineType(types: !44)
!44 = !{!45, !46}
!45 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!46 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !47, size: 64)
!47 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !48)
!48 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!49 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !50, file: !14, line: 151)
!50 = !DISubprogram(name: "atoi", scope: !6, file: !6, line: 105, type: !51, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!51 = !DISubroutineType(types: !52)
!52 = !{!9, !46}
!53 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !54, file: !14, line: 152)
!54 = !DISubprogram(name: "atol", scope: !6, file: !6, line: 108, type: !55, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!55 = !DISubroutineType(types: !56)
!56 = !{!20, !46}
!57 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !58, file: !14, line: 153)
!58 = !DISubprogram(name: "bsearch", scope: !6, file: !6, line: 960, type: !59, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!59 = !DISubroutineType(types: !60)
!60 = !{!30, !61, !61, !31, !31, !63}
!61 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !62, size: 64)
!62 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!63 = !DIDerivedType(tag: DW_TAG_typedef, name: "__compar_fn_t", file: !6, line: 948, baseType: !64)
!64 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !65, size: 64)
!65 = !DISubroutineType(types: !66)
!66 = !{!9, !61, !61}
!67 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !68, file: !14, line: 154)
!68 = !DISubprogram(name: "calloc", scope: !6, file: !6, line: 675, type: !28, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!69 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !70, file: !14, line: 155)
!70 = !DISubprogram(name: "div", scope: !6, file: !6, line: 998, type: !71, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!71 = !DISubroutineType(types: !72)
!72 = !{!12, !9, !9}
!73 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !74, file: !14, line: 156)
!74 = !DISubprogram(name: "exit", scope: !6, file: !6, line: 756, type: !75, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagOptimized)
!75 = !DISubroutineType(types: !76)
!76 = !{null, !9}
!77 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !78, file: !14, line: 157)
!78 = !DISubprogram(name: "free", scope: !6, file: !6, line: 687, type: !79, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!79 = !DISubroutineType(types: !80)
!80 = !{null, !30}
!81 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !82, file: !14, line: 158)
!82 = !DISubprogram(name: "getenv", scope: !6, file: !6, line: 773, type: !83, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!83 = !DISubroutineType(types: !84)
!84 = !{!85, !46}
!85 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !48, size: 64)
!86 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !87, file: !14, line: 159)
!87 = !DISubprogram(name: "labs", scope: !6, file: !6, line: 981, type: !88, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!88 = !DISubroutineType(types: !89)
!89 = !{!20, !20}
!90 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !91, file: !14, line: 160)
!91 = !DISubprogram(name: "ldiv", scope: !6, file: !6, line: 1000, type: !92, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!92 = !DISubroutineType(types: !93)
!93 = !{!16, !20, !20}
!94 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !95, file: !14, line: 161)
!95 = !DISubprogram(name: "malloc", scope: !6, file: !6, line: 672, type: !96, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!96 = !DISubroutineType(types: !97)
!97 = !{!30, !31}
!98 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !99, file: !14, line: 163)
!99 = !DISubprogram(name: "mblen", scope: !6, file: !6, line: 1068, type: !100, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!100 = !DISubroutineType(types: !101)
!101 = !{!9, !46, !31}
!102 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !103, file: !14, line: 164)
!103 = !DISubprogram(name: "mbstowcs", scope: !6, file: !6, line: 1079, type: !104, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!104 = !DISubroutineType(types: !105)
!105 = !{!31, !106, !109, !31}
!106 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !107)
!107 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !108, size: 64)
!108 = !DIBasicType(name: "wchar_t", size: 32, encoding: DW_ATE_signed)
!109 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !46)
!110 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !111, file: !14, line: 165)
!111 = !DISubprogram(name: "mbtowc", scope: !6, file: !6, line: 1071, type: !112, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!112 = !DISubroutineType(types: !113)
!113 = !{!9, !106, !109, !31}
!114 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !115, file: !14, line: 167)
!115 = !DISubprogram(name: "qsort", scope: !6, file: !6, line: 970, type: !116, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!116 = !DISubroutineType(types: !117)
!117 = !{null, !30, !31, !31, !63}
!118 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !119, file: !14, line: 170)
!119 = !DISubprogram(name: "quick_exit", scope: !6, file: !6, line: 762, type: !75, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagOptimized)
!120 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !121, file: !14, line: 173)
!121 = !DISubprogram(name: "rand", scope: !6, file: !6, line: 573, type: !122, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!122 = !DISubroutineType(types: !123)
!123 = !{!9}
!124 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !125, file: !14, line: 174)
!125 = !DISubprogram(name: "realloc", scope: !6, file: !6, line: 683, type: !126, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!126 = !DISubroutineType(types: !127)
!127 = !{!30, !30, !31}
!128 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !129, file: !14, line: 175)
!129 = !DISubprogram(name: "srand", scope: !6, file: !6, line: 575, type: !130, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!130 = !DISubroutineType(types: !131)
!131 = !{null, !132}
!132 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!133 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !134, file: !14, line: 176)
!134 = !DISubprogram(name: "strtod", scope: !6, file: !6, line: 118, type: !135, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!135 = !DISubroutineType(types: !136)
!136 = !{!45, !109, !137}
!137 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !138)
!138 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !85, size: 64)
!139 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !140, file: !14, line: 177)
!140 = !DISubprogram(name: "strtol", linkageName: "__isoc23_strtol", scope: !6, file: !6, line: 215, type: !141, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!141 = !DISubroutineType(types: !142)
!142 = !{!20, !109, !137, !9}
!143 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !144, file: !14, line: 178)
!144 = !DISubprogram(name: "strtoul", linkageName: "__isoc23_strtoul", scope: !6, file: !6, line: 219, type: !145, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!145 = !DISubroutineType(types: !146)
!146 = !{!33, !109, !137, !9}
!147 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !148, file: !14, line: 179)
!148 = !DISubprogram(name: "system", scope: !6, file: !6, line: 923, type: !51, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!149 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !150, file: !14, line: 181)
!150 = !DISubprogram(name: "wcstombs", scope: !6, file: !6, line: 1083, type: !151, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!151 = !DISubroutineType(types: !152)
!152 = !{!31, !153, !154, !31}
!153 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !85)
!154 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !155)
!155 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !156, size: 64)
!156 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !108)
!157 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !158, file: !14, line: 182)
!158 = !DISubprogram(name: "wctomb", scope: !6, file: !6, line: 1075, type: !159, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!159 = !DISubroutineType(types: !160)
!160 = !{!9, !85, !108}
!161 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !162, entity: !163, file: !14, line: 210)
!162 = !DINamespace(name: "__gnu_cxx", scope: null)
!163 = !DIDerivedType(tag: DW_TAG_typedef, name: "lldiv_t", file: !6, line: 81, baseType: !164)
!164 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !6, line: 77, size: 128, flags: DIFlagTypePassByValue, elements: !165, identifier: "_ZTS7lldiv_t")
!165 = !{!166, !168}
!166 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !164, file: !6, line: 79, baseType: !167, size: 64)
!167 = !DIBasicType(name: "long long", size: 64, encoding: DW_ATE_signed)
!168 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !164, file: !6, line: 80, baseType: !167, size: 64, offset: 64)
!169 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !162, entity: !170, file: !14, line: 216)
!170 = !DISubprogram(name: "_Exit", scope: !6, file: !6, line: 768, type: !75, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagOptimized)
!171 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !162, entity: !172, file: !14, line: 222)
!172 = !DISubprogram(name: "llabs", scope: !6, file: !6, line: 984, type: !173, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!173 = !DISubroutineType(types: !174)
!174 = !{!167, !167}
!175 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !162, entity: !176, file: !14, line: 228)
!176 = !DISubprogram(name: "lldiv", scope: !6, file: !6, line: 1004, type: !177, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!177 = !DISubroutineType(types: !178)
!178 = !{!163, !167, !167}
!179 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !162, entity: !180, file: !14, line: 240)
!180 = !DISubprogram(name: "atoll", scope: !6, file: !6, line: 113, type: !181, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!181 = !DISubroutineType(types: !182)
!182 = !{!167, !46}
!183 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !162, entity: !184, file: !14, line: 241)
!184 = !DISubprogram(name: "strtoll", linkageName: "__isoc23_strtoll", scope: !6, file: !6, line: 238, type: !185, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!185 = !DISubroutineType(types: !186)
!186 = !{!167, !109, !137, !9}
!187 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !162, entity: !188, file: !14, line: 242)
!188 = !DISubprogram(name: "strtoull", linkageName: "__isoc23_strtoull", scope: !6, file: !6, line: 243, type: !189, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!189 = !DISubroutineType(types: !190)
!190 = !{!191, !109, !137, !9}
!191 = !DIBasicType(name: "unsigned long long", size: 64, encoding: DW_ATE_unsigned)
!192 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !162, entity: !193, file: !14, line: 244)
!193 = !DISubprogram(name: "strtof", scope: !6, file: !6, line: 124, type: !194, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!194 = !DISubroutineType(types: !195)
!195 = !{!196, !109, !137}
!196 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!197 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !162, entity: !198, file: !14, line: 245)
!198 = !DISubprogram(name: "strtold", scope: !6, file: !6, line: 127, type: !199, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!199 = !DISubroutineType(types: !200)
!200 = !{!201, !109, !137}
!201 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!202 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !163, file: !14, line: 253)
!203 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !170, file: !14, line: 255)
!204 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !172, file: !14, line: 257)
!205 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !206, file: !14, line: 258)
!206 = !DISubprogram(name: "div", linkageName: "_ZN9__gnu_cxx3divExx", scope: !162, file: !14, line: 225, type: !177, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!207 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !176, file: !14, line: 259)
!208 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !180, file: !14, line: 261)
!209 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !193, file: !14, line: 262)
!210 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !184, file: !14, line: 263)
!211 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !188, file: !14, line: 264)
!212 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !198, file: !14, line: 265)
!213 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !23, file: !214, line: 38)
!214 = !DIFile(filename: "/usr/lib64/gcc/x86_64-pc-linux-gnu/15.2.1/../../../../include/c++/15.2.1/stdlib.h", directory: "", checksumkind: CSK_MD5, checksum: "6b5a21b1805b4429608f31a862826533")
!215 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !35, file: !214, line: 39)
!216 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !74, file: !214, line: 40)
!217 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !40, file: !214, line: 43)
!218 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !119, file: !214, line: 46)
!219 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !170, file: !214, line: 49)
!220 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !12, file: !214, line: 54)
!221 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !16, file: !214, line: 55)
!222 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !223, file: !214, line: 57)
!223 = !DISubprogram(name: "abs", linkageName: "_ZSt3absg", scope: !4, file: !10, line: 143, type: !224, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!224 = !DISubroutineType(types: !225)
!225 = !{!226, !226}
!226 = !DIBasicType(name: "__float128", size: 128, encoding: DW_ATE_float)
!227 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !42, file: !214, line: 58)
!228 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !50, file: !214, line: 59)
!229 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !54, file: !214, line: 60)
!230 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !58, file: !214, line: 61)
!231 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !68, file: !214, line: 62)
!232 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !206, file: !214, line: 63)
!233 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !78, file: !214, line: 64)
!234 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !82, file: !214, line: 65)
!235 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !87, file: !214, line: 66)
!236 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !91, file: !214, line: 67)
!237 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !95, file: !214, line: 68)
!238 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !99, file: !214, line: 70)
!239 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !103, file: !214, line: 71)
!240 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !111, file: !214, line: 72)
!241 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !115, file: !214, line: 74)
!242 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !121, file: !214, line: 75)
!243 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !125, file: !214, line: 76)
!244 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !129, file: !214, line: 77)
!245 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !134, file: !214, line: 78)
!246 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !140, file: !214, line: 79)
!247 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !144, file: !214, line: 80)
!248 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !148, file: !214, line: 81)
!249 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !150, file: !214, line: 83)
!250 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !158, file: !214, line: 84)
!251 = !{i32 7, !"Dwarf Version", i32 5}
!252 = !{i32 2, !"Debug Info Version", i32 3}
!253 = !{i32 1, !"wchar_size", i32 4}
!254 = !{i32 8, !"PIC Level", i32 2}
!255 = !{i32 7, !"PIE Level", i32 2}
!256 = !{i32 7, !"uwtable", i32 2}
!257 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!258 = !{!"clang version 21.1.3 (git@github.com:tarunprabhu/kitsune.git c21cc3bd73b905422e733554fab8dcc1781c85dd)"}
!259 = distinct !DISubprogram(name: "vecadd", scope: !260, file: !260, line: 3, type: !261, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !264)
!260 = !DIFile(filename: "/tmp/test.cpp", directory: "", checksumkind: CSK_MD5, checksum: "8aac604d479ae91586e56a9af3432d5d")
!261 = !DISubroutineType(types: !262)
!262 = !{null, !263, !263, !263, !20}
!263 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!264 = !{!265, !266, !267, !268, !269, !271}
!265 = !DILocalVariable(name: "c", arg: 1, scope: !259, file: !260, line: 3, type: !263)
!266 = !DILocalVariable(name: "a", arg: 2, scope: !259, file: !260, line: 3, type: !263)
!267 = !DILocalVariable(name: "b", arg: 3, scope: !259, file: !260, line: 3, type: !263)
!268 = !DILocalVariable(name: "n", arg: 4, scope: !259, file: !260, line: 3, type: !20)
!269 = !DILocalVariable(name: "i", scope: !270, file: !260, line: 4, type: !20)
!270 = distinct !DILexicalBlock(scope: !259, file: !260, line: 4, column: 3)
!271 = !DILocalVariable(name: "i", scope: !272, file: !260, line: 4, type: !20)
!272 = distinct !DILexicalBlock(scope: !270, file: !260, line: 4, column: 3)
!273 = !DILocation(line: 0, scope: !259)
!274 = !DILocation(line: 0, scope: !270)
!275 = !DILocation(line: 4, column: 25, scope: !272)
!276 = !DILocation(line: 4, column: 3, scope: !270)
!277 = !DILocation(line: 0, scope: !272)
!278 = !DILocation(line: 5, column: 12, scope: !279)
!279 = distinct !DILexicalBlock(scope: !272, file: !260, line: 4, column: 35)
!280 = !{!281, !281, i64 0}
!281 = !{!"int", !282, i64 0}
!282 = !{!"omnipotent char", !283, i64 0}
!283 = !{!"Simple C++ TBAA"}
!284 = !DILocation(line: 5, column: 19, scope: !279)
!285 = !DILocation(line: 5, column: 17, scope: !279)
!286 = !DILocation(line: 5, column: 5, scope: !279)
!287 = !DILocation(line: 5, column: 10, scope: !279)
!288 = !DILocation(line: 6, column: 3, scope: !279)
!289 = !DILocation(line: 4, column: 31, scope: !272)
!290 = distinct !{!290, !276, !291, !292, !314, !315, !316, !317}
!291 = !DILocation(line: 6, column: 3, scope: !270)
!292 = !{!"tapir.loop.lowering.enabled"}
!293 = !DILocation(line: 4, column: 3, scope: !272)
!294 = !DILocation(line: 7, column: 1, scope: !259)
!295 = distinct !DISubprogram(name: "f", scope: !260, file: !260, line: 9, type: !296, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !298)
!296 = !DISubroutineType(types: !297)
!297 = !{null, !263, !263, !20}
!298 = !{!299, !300, !301}
!299 = !DILocalVariable(name: "c", arg: 1, scope: !295, file: !260, line: 9, type: !263)
!300 = !DILocalVariable(name: "a", arg: 2, scope: !295, file: !260, line: 9, type: !263)
!301 = !DILocalVariable(name: "n", arg: 3, scope: !295, file: !260, line: 9, type: !20)
!302 = !DILocation(line: 0, scope: !295)
!303 = !DILocation(line: 10, column: 3, scope: !295)
!304 = !DILocation(line: 11, column: 1, scope: !295)
!305 = distinct !DISubprogram(name: "g", scope: !260, file: !260, line: 13, type: !261, scopeLine: 13, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !306)
!306 = !{!307, !308, !309, !310}
!307 = !DILocalVariable(name: "c", arg: 1, scope: !305, file: !260, line: 13, type: !263)
!308 = !DILocalVariable(name: "a", arg: 2, scope: !305, file: !260, line: 13, type: !263)
!309 = !DILocalVariable(name: "b", arg: 3, scope: !305, file: !260, line: 13, type: !263)
!310 = !DILocalVariable(name: "n", arg: 4, scope: !305, file: !260, line: 13, type: !20)
!311 = !DILocation(line: 0, scope: !305)
!312 = !DILocation(line: 14, column: 3, scope: !305)
!313 = !DILocation(line: 15, column: 1, scope: !305)
!314 = !{!"tapir.loop.spawn.strategy", i32 3}
!315 = !{!"tapir.loop.target", i32 4}
!316 = !{!"tapir.loop.perfect.depth", i32 1}
!317 = !{!"tapir.loop.perfect.level", i32 1}
