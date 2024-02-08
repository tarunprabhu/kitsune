//===- InstrumentationTest.cpp - Unit tests for instrumentation utilities -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/Instrumentation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormatVariadic.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

static void checkInsertPrintf(Function &f, StringRef stream) {
  std::string buf;
  raw_string_ostream os(buf);
  os << f << "\n";
  os.flush();

  SmallVector<StringRef, 8> lines;
  SplitString(buf, lines, "\n\r");

  SmallVector<std::string, 8> expected = {
      "define void @f() {",
      llvm::formatv("  %1 = load ptr, ptr @{0}", stream),
      "  %2 = call i32 (ptr, ptr, ...) @fprintf(ptr %1, ptr @0, i32 21)",
      "  ret void",
      "}",
  };

  EXPECT_EQ(lines.size(), expected.size());
  for (unsigned i = 0, e = lines.size(); i < e; ++i)
    EXPECT_TRUE(lines[i].starts_with(expected[i]));
}

#define TEST_INSERT_PRINTF_HEADER                                              \
  LLVMContext ctx;                                                             \
  Module m("", ctx);                                                           \
                                                                               \
  Type *voidTy = Type::getVoidTy(ctx);                                         \
  Type *i32 = Type::getInt32Ty(ctx);                                           \
  FunctionType *fty = FunctionType::get(voidTy, {}, /*IsVarArg=*/false);       \
                                                                               \
  Constant *c = ConstantInt::get(i32, 21, /*isSigned=*/true);                  \
  Function *f = Function::Create(fty, GlobalValue::ExternalLinkage, "f", m);   \
  BasicBlock *bb = BasicBlock::Create(ctx, "", f);                             \
  ReturnInst *ret = ReturnInst::Create(ctx, bb);

TEST(KitInstrumentation, insertPrintfStdoutBuilder) {
  TEST_INSERT_PRINTF_HEADER

  IRBuilder<> builder(ret);
  insertPrintStdout(builder, "write: %d\n", {c});
  checkInsertPrintf(*f, "stdout");
}

TEST(KitInstrumentation, insertPrintfStderrBuilder) {
  TEST_INSERT_PRINTF_HEADER

  IRBuilder<> builder(ret);
  insertPrintStderr(builder, "write: %d\n", {c});
  checkInsertPrintf(*f, "stderr");
}

TEST(KitInstrumentation, insertPrintfStdoutInsertPt) {
  TEST_INSERT_PRINTF_HEADER

  insertPrintStdout(ret->getIterator(), "write: %d\n", {c});
  checkInsertPrintf(*f, "stdout");
}

TEST(KitInstrumentation, insertPrintfStderrInsertPt) {
  TEST_INSERT_PRINTF_HEADER

  insertPrintStderr(ret->getIterator(), "write: %d\n", {c});
  checkInsertPrintf(*f, "stderr");
}

} // namespace
