//==- InstAttrsTest.cpp - Unit tests for Kitsune's instruction attributes --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/InstAttrs.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

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
#define INST_ATTR(NAME, TYPE, IRNAME)                                          \
  EXPECT_EQ(getAttrName(InstAttrKind::NAME), IRNAME);                          \
  EXPECT_TRUE(getAttrName(InstAttrKind::NAME).starts_with("kit.inst."));
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attrKind) {
  EXPECT_EQ(getInstAttrKind("whoops"), std::nullopt);

#define INST_ATTR(NAME, TYPE, IRNAME)                                          \
  EXPECT_EQ(getInstAttrKind(IRNAME), InstAttrKind::NAME);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

#define CHECK_GENERIC_FLAG(NAME, INST)                                         \
  EXPECT_FALSE(hasAttr(INST, InstAttrKind::NAME));                             \
  addAttr(INST, InstAttrKind::NAME);                                           \
  EXPECT_TRUE(hasAttr(INST, InstAttrKind::NAME));                              \
  removeAttr(INST, InstAttrKind::NAME);                                        \
  EXPECT_FALSE(hasAttr(INST, InstAttrKind::NAME));

#define CHECK_GENERIC(NAME, INST, VAL)                                         \
  EXPECT_FALSE(hasAttr(INST, InstAttrKind::NAME));                             \
  add##NAME##Attr(INST, VAL);                                                  \
  EXPECT_TRUE(hasAttr(INST, InstAttrKind::NAME));                              \
  removeAttr(INST, InstAttrKind::NAME);                                        \
  EXPECT_FALSE(hasAttr(INST, InstAttrKind::NAME));                             \
  EXPECT_EXIT(addAttr(INST, InstAttrKind::NAME), ::testing::ExitedWithCode(1), \
              "error: cannot add attribute");

TEST(KitInstAttrs, attrsGeneric) {
  LLVMContext ctx;
  std::unique_ptr<Module> m = parseIR(ctx, ll);
  Function *f = m->getFunction("f");
  DominatorTree dt(*f);
  LoopInfo li(dt);
  Loop *loop = *li.begin();
  [[maybe_unused]] Instruction *inst = loop->getLatchCmpInst();

#define INST_ATTR_FLAG(NAME, IRNAME) CHECK_GENERIC_FLAG(NAME, *inst)
#define INST_ATTR_ENUM(NAME, IRNAME, TYPE)                                \
  CHECK_GENERIC(NAME, *inst, (TYPE)1)
#define INST_ATTR_F32(NAME, IRNAME) CHECK_GENERIC(NAME, *inst, 3.14F)
#define INST_ATTR_F64(NAME, IRNAME) CHECK_GENERIC(NAME, *inst, 3.141592653)
#define INST_ATTR_I32(NAME, IRNAME) CHECK_GENERIC(NAME, *inst, 67)
#define INST_ATTR_I64(NAME, IRNAME) CHECK_GENERIC(NAME, *inst, 73L)
#define INST_ATTR_STR(NAME, IRNAME) CHECK_GENERIC(NAME, *inst, "queen's")
#define INST_ATTR_LOOP(NAME, IRNAME) CHECK_GENERIC(NAME, *inst, *loop)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, flagAttrs) {
  LLVMContext ctx;
  [[maybe_unused]] ReturnInst *inst = ReturnInst::Create(ctx);

#define INST_ATTR_FLAG(NAME, IRNAME)                                      \
  EXPECT_FALSE(has##NAME##Attr(INST));                                         \
                                                                               \
  add##NAME##Attr(INST);                                                       \
  EXPECT_TRUE(has##NAME##Attr(INST));                                          \
                                                                               \
  add##NAME##Attr(INST);                                                       \
  EXPECT_TRUE(has##NAME##Attr(INST));                                          \
                                                                               \
  remove##NAME##Attr(INST);                                                    \
  EXPECT_FALSE(has##NAME##Attr(INST));

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

#define CHECK_ACCESSORS(NAME, INST, VAL1, VAL2)                                \
  EXPECT_FALSE(has##NAME##Attr(INST));                                         \
                                                                               \
  add##NAME##Attr(INST, VAL1);                                                 \
  EXPECT_TRUE(has##NAME##Attr(INST));                                          \
  EXPECT_EQ(get##NAME##Attr(INST), (VAL1));                                    \
                                                                               \
  add##NAME##Attr(INST, VAL2);                                                 \
  EXPECT_TRUE(has##NAME##Attr(INST));                                          \
  EXPECT_EQ(get##NAME##Attr(INST), (VAL2));                                    \
                                                                               \
  remove##NAME##Attr(INST);                                                    \
  EXPECT_FALSE(has##NAME##Attr(INST));

TEST(KitInstAttrs, enumAttrs) {
  LLVMContext ctx;
  [[maybe_unused]] ReturnInst *inst = ReturnInst::Create(ctx);

  // WARNING: This is somewhat risky because there is no guarantee that the
  // integer values 1 and 2 will be valid for every enum type that we may have
  // an attribute for. If this ever happens, change this test. It may be
  // sufficient to just check for some enum-valued attribute instead of all of
  // them.
#define INST_ATTR_ENUM(NAME, IRNAME, TYPE)                                \
  CHECK_ACCESSORS(NAME, *inst, (TYPE)1, (TYPE)2)

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, f32Test) {
  LLVMContext ctx;
  [[maybe_unused]] ReturnInst *inst = ReturnInst::Create(ctx);

#define INST_ATTR_F32(NAME, IRNAME)                                       \
  CHECK_ACCESSORS(NAME, *inst, 3.14159F, 2.71828F)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, f64Test) {
  LLVMContext ctx;
  [[maybe_unused]] ReturnInst *inst = ReturnInst::Create(ctx);

#define INST_ATTR_F64(NAME, IRNAME)                                       \
  CHECK_ACCESSORS(NAME, *inst, 3.1415926535, 2.71828)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, i32Test) {
  LLVMContext ctx;
  [[maybe_unused]] ReturnInst *inst = ReturnInst::Create(ctx);

#define INST_ATTR_I32(NAME, IRNAME) CHECK_ACCESSORS(NAME, *inst, 42, 97)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, i64Test) {
  LLVMContext ctx;
  [[maybe_unused]] ReturnInst *inst = ReturnInst::Create(ctx);

#define INST_ATTR_I64(NAME, IRNAME) CHECK_ACCESSORS(NAME, *inst, 2L, 7L)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, loopTest) {
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

#define INST_ATTR_LOOP(NAME, IRNAME)                                      \
  EXPECT_FALSE(has##NAME##Attr(*inst));                                        \
                                                                               \
  add##NAME##Attr(*inst, loopF);                                               \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##NAME##Attr(*inst, lis), loopF);                               \
                                                                               \
  add##NAME##Attr(*inst, *loopG);                                              \
  EXPECT_TRUE(has##NAME##Attr(*inst));                                         \
  EXPECT_EQ(get##NAME##Attr(*inst, lis), loopG);                               \
                                                                               \
  remove##NAME##Attr(*inst);                                                   \
  EXPECT_FALSE(has##NAME##Attr(*inst));

#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, strTest) {
  LLVMContext ctx;
  [[maybe_unused]] ReturnInst *inst = ReturnInst::Create(ctx);

#define INST_ATTR_STR(NAME, IRNAME)                                       \
  CHECK_ACCESSORS(NAME, *inst, "magdalen", "exeter")
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

} // namespace
