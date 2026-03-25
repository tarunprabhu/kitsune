//==- InstAttrsTest.cpp - Unit tests for Kitsune's instruction attributes --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/InstAttrs.h"
#include "TestAttrsCommon.h"
#include "TestValues.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

// The standard accessors do not allow us to create invalid attributes. To
// create one, we have to know how these are added to the function. This is not
// unreasonable since the create functions are a fairly thin wrappers around
// LLVM's existing support.
static void addMetadata(Instruction &inst, StringRef name,
                        ArrayRef<Metadata *> ops) {
  LLVMContext &ctx = inst.getContext();
  inst.setMetadata(name, MDNode::get(ctx, ops));
}

// Create metadata consisting of `n` "empty" operands.
static void addMetadata(Instruction &inst, StringRef name, unsigned n) {
  LLVMContext &ctx = inst.getContext();
  MDNode *mdEmpty = MDNode::get(ctx, {});
  SmallVector<Metadata *, 8> ops;

  ops.append(n, mdEmpty);
  addMetadata(inst, name, ops);
}

static constexpr StringRef ll = R"(
define void @f(i64 %n) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.header ]
  %inc.i = add i64 %i, 1
  %exitcond.i.not = icmp eq i64 %inc.i, %n
  br i1 %exitcond.i.not, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  ret void
}

define void @g(i64 %n) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.header ]
  %inc.i = add i64 %i, 1
  %exitcond.i.not = icmp eq i64 %inc.i, %n
  br i1 %exitcond.i.not, label %for.i.exit, label %for.i.header, !llvm.loop !1

for.i.exit:
  ret void
}

!0 = distinct !{!0}
!1 = distinct !{!1}
)";

static std::unique_ptr<Module> parseIR(LLVMContext &ctx, StringRef ir) {
  SMDiagnostic err;
  std::unique_ptr<Module> m = parseAssemblyString(ir, err, ctx);
  if (!m)
    err.print("parseIR", errs());
  return m;
}

TEST(KitInstAttrs, attrName) {
#define INST_ATTR(NAME, IRNAME, TYPE)                                          \
  EXPECT_EQ(getAttrName(InstAttrKind::NAME), IRNAME);                          \
  EXPECT_TRUE(getAttrName(InstAttrKind::NAME).starts_with("kit.inst."));
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attrKind) {
  EXPECT_EQ(getInstAttrKind("queen's"), std::nullopt);
#define INST_ATTR(NAME, IRNAME, TYPE)                                          \
  EXPECT_EQ(getInstAttrKind(IRNAME), InstAttrKind::NAME);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

#define DECLS(OS, OBJ)                                                         \
  std::string buf;                                                             \
  raw_string_ostream OS(buf);                                                  \
  LLVMContext ctx;                                                             \
  [[maybe_unused]] ReturnInst OBJ = ReturnInst::Create(ctx);

TEST(KitInstAttrs, verifyGeneric) {
  DECLS(os, *inst);
#define INST_ATTR_0(NAME, IRNAME)                                              \
  TEST_GENERIC_VERIFY_0(os, *inst, InstAttrKind, NAME, IRNAME);
#define INST_ATTR(NAME, IRNAME, TYPE)                                          \
  TEST_GENERIC_VERIFY_N(os, *inst, InstAttrKind, NAME, IRNAME);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attrsGeneric) {
  DECLS(os, *inst);

#define INST_ATTR_0(NAME, IRNAME)                                              \
  TEST_GENERIC_ATTR_0(*inst, InstAttrKind, NAME);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_0(NAME, IRNAME)
#define INST_ATTR(NAME, IRNAME, TYPE)                                          \
  TEST_GENERIC_ATTR_N(*inst, InstAttrKind, NAME);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr0) {
  DECLS(os, *inst);
#define INST_ATTR_0(NAME, IRNAME) TEST_ATTR_0(os, *inst, NAME, IRNAME);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr1) {
  DECLS(os, *inst);
#define INST_ATTR_1(NAME, IRNAME, TYPE)                                        \
  TEST_ATTR_1(os, *inst, NAME, IRNAME, TYPE);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr2) {
  DECLS(os, *inst);
#define INST_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)        \
  TEST_ATTR_2(os, *inst, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr3) {
  DECLS(os, *inst);
#define INST_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2)                                               \
  TEST_ATTR_3(os, *inst, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,   \
              ETY2, ENAME2, EN2);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr4) {
  DECLS(os, *inst);
#define INST_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3)                            \
  TEST_ATTR_4(os, *inst, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,   \
              ETY2, ENAME2, EN2, ETY3, ENAME3, EN3);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr5) {
  DECLS(os, *inst);
#define INST_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)         \
  TEST_ATTR_5(os, *inst, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,   \
              ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr6) {
  DECLS(os, *inst);
#define INST_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5)                                               \
  TEST_ATTR_6(os, *inst, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,   \
              ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
              ENAME5, EN5);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr7) {
  DECLS(os, *inst);
#define INST_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6)                            \
  TEST_ATTR_7(os, *inst, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,   \
              ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
              ENAME5, EN5, ETY6, ENAME6, EN6);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr8) {
  DECLS(os, *inst);
#define INST_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)         \
  TEST_ATTR_8(os, *inst, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,   \
              ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
              ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, loop) {
  std::string buf;
  raw_string_ostream os(buf);
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dtf(*f);
  LoopInfo lif(dtf);
  Function *g = m->getFunction("g");
  DominatorTree dtg(*g);
  LoopInfo lig(dtg);
  SmallVector<const LoopInfo *, 4> lis = {&lig, &lif};
  [[maybe_unused]] Loop *loopF = *lif.begin();
  [[maybe_unused]] Loop *loopG = *lig.begin();
  [[maybe_unused]] Instruction *inst = ReturnInst::Create(ctx);

#define INST_ATTR_LOOP(NAME, IRNAME)                                           \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
                                                                               \
  add##NAME##Attr(*inst, loopF);                                               \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##NAME##Attr(*inst, lis), loopF);                               \
                                                                               \
  add##NAME##Attr(*inst, *loopG);                                              \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##NAME##Attr(*inst, lis), loopG);                               \
                                                                               \
  remove##NAME##Attr(*inst);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
  EXPECT_TRUE(verify##NAME##Attr(*inst));                                      \
                                                                               \
  os.str().clear();                                                            \
  addMetadata(*inst, IRNAME, 1);                                               \
  EXPECT_FALSE(verify##NAME##Attr(*inst));                                     \
  EXPECT_FALSE(verify##NAME##Attr(*inst, &os));                                \
  EXPECT_TRUE(StringRef(os.str()).starts_with("Missing value of type"));       \
  remove##NAME##Attr(*inst);

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

} // namespace
