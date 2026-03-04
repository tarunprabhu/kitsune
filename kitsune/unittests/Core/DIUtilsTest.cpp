//===- DIUtilsTest.cpp - Unit tests for debug info utilities --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/DIUtils.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &ctx, StringRef ir) {
  SMDiagnostic err;
  std::unique_ptr<Module> m = parseAssemblyString(ir, err, ctx);
  if (!m)
    err.print("parseIR", errs());
  return m;
}

template <typename InstType>
static const DebugLoc getDebugLocFor(const Function &f) {
  for (const_inst_iterator i = inst_begin(f), ie = inst_end(f); i != ie; ++i)
    if (auto *asType = dyn_cast<InstType>(&*i))
      return asType->getDebugLoc();
  return DebugLoc();
}

TEST(DIUtils, toString) {
  constexpr StringRef ir = R"(
target triple = "x86_64-pc-linux-gnu"

define i32 @f(i32 %0) !dbg !10 {
    #dbg_value(i32 %0, !16, !DIExpression(), !17)
    #dbg_value(i32 %0, !18, !DIExpression(), !21)
  %2 = add nsw i32 %0, 1, !dbg !23
  %3 = load i32, ptr null, !dbg !25
  ret i32 %2, !dbg !24
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 21.1.6", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/tmp/inlined.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 21.1.6"}
!10 = distinct !DISubprogram(name: "f", scope: !11, file: !11, line: 5, type: !12, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !15)
!11 = !DIFile(filename: "/tmp/inlined.c", directory: "", checksumkind: CSK_MD5, checksum: "d6d534772b9393479ad01e0b69750454")
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{!16}
!16 = !DILocalVariable(name: "n", arg: 1, scope: !10, file: !11, line: 5, type: !14)
!17 = !DILocation(line: 0, scope: !10)
!18 = !DILocalVariable(name: "n", arg: 1, scope: !19, file: !11, line: 1, type: !14)
!19 = distinct !DISubprogram(name: "add1", scope: !11, file: !11, line: 1, type: !12, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !20)
!20 = !{!18}
!21 = !DILocation(line: 0, scope: !19, inlinedAt: !22)
!22 = distinct !DILocation(line: 6, column: 10, scope: !10)
!23 = !DILocation(line: 2, column: 12, scope: !19, inlinedAt: !22)
!24 = !DILocation(line: 6, column: 3, scope: !10)
!25 = !DILocation(line: 42, scope: !10)
)";

  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ir);
  Function *f = m->getFunction("f");

  DebugLoc invLoc = DebugLoc();
  EXPECT_EQ(toString(invLoc), "");
  EXPECT_EQ(toString(invLoc, /*inlinedAt=*/true), "");

  DebugLoc addLoc = getDebugLocFor<BinaryOperator>(*f);
  EXPECT_EQ(toString(addLoc), "/tmp/inlined.c:2:12");
  EXPECT_EQ(toString(addLoc, /*inlinedAt=*/false), "/tmp/inlined.c:2:12");
  EXPECT_EQ(toString(addLoc, /*inlinedAt=*/true),
            "/tmp/inlined.c:2:12@[/tmp/inlined.c:6:10]");

  DebugLoc retLoc = getDebugLocFor<ReturnInst>(*f);
  EXPECT_EQ(toString(retLoc), "/tmp/inlined.c:6:3");
  EXPECT_EQ(toString(retLoc, /*inlinedAt=*/false), "/tmp/inlined.c:6:3");
  EXPECT_EQ(toString(retLoc, /*inlinedAt=*/true), "/tmp/inlined.c:6:3");

  DebugLoc loadLoc = getDebugLocFor<LoadInst>(*f);
  EXPECT_EQ(toString(loadLoc), "/tmp/inlined.c:42");
  EXPECT_EQ(toString(loadLoc, /*inlinedAt=*/false), "/tmp/inlined.c:42");
  EXPECT_EQ(toString(loadLoc, /*inlinedAt=*/true), "/tmp/inlined.c:42");
}
