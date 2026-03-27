//==- InstAttrsTest.cpp - Unit tests for Kitsune's instruction attributes --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/InstAttrs.h"
#include "TestAttrsCommon.h"
#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Core/InstUtils.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

static void addMetadata(Instruction &inst, StringRef attrName,
                        ArrayRef<Metadata *> attrVals) {
  LLVMContext &ctx = inst.getContext();
  MDNode *attrList = getAttrList(inst);
  MDNode *newAttrList = getAttrListWith(attrName, attrVals, attrList, ctx);

  inst.setMetadata(LLVMContext::MD_kit_inst_attrs, newAttrList);
}

// Create metadata consisting of `n` "empty" operands.
static void addMetadata(Instruction &inst, StringRef attrName, unsigned n) {
  LLVMContext &ctx = inst.getContext();
  MDNode *mdEmpty = MDNode::get(ctx, {});
  SmallVector<Metadata *, 8> attrVals;

  attrVals.append(n, mdEmpty);
  addMetadata(inst, attrName, attrVals);
}

TEST(KitInstAttrs, attrName) {
#define INST_ATTR(NAME, IRNAME, ...)                                           \
  EXPECT_EQ(getAttrName(InstAttrKind::NAME), IRNAME);                          \
  EXPECT_TRUE(getAttrName(InstAttrKind::NAME).starts_with("kit.inst."));
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attrKind) {
  EXPECT_EQ(getInstAttrKind("queen's"), std::nullopt);
#define INST_ATTR(NAME, IRNAME, ...)                                           \
  EXPECT_EQ(getInstAttrKind(IRNAME), InstAttrKind::NAME);
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

#define DECLS(OS, OBJ)                                                         \
  std::string buf;                                                             \
  raw_string_ostream OS(buf);                                                  \
  LLVMContext ctx;                                                             \
  Type *voidTy = Type::getVoidTy(ctx);                                         \
  FunctionType *fty = FunctionType::get(voidTy, {}, /*IsVarArg=*/false);       \
  Module m("", ctx);                                                           \
  Function *f = cast<Function>(m.getOrInsertFunction("f", fty).getCallee());   \
  BasicBlock *bb = BasicBlock::Create(ctx, "entry", f);                        \
  [[maybe_unused]] ReturnInst OBJ =                                            \
      ReturnInst::Create(ctx, /*retVal=*/nullptr, bb->getFirstInsertionPt());

TEST(KitInstAttrs, verifyGeneric) {
  DECLS(os, *inst);
#define INST_ATTR_0(NAME, IRNAME, ...)                                         \
  TEST_GENERIC_VERIFY_0(os, *inst, InstAttrKind, NAME, IRNAME)
#define INST_ATTR(NAME, IRNAME, ...)                                           \
  TEST_GENERIC_VERIFY_N(os, *inst, InstAttrKind, NAME, IRNAME)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attrsGeneric) {
  DECLS(os, *inst);

#define INST_ATTR_0(NAME, ...) TEST_GENERIC_ATTR_0(*inst, InstAttrKind, NAME)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_0(...)
#define INST_ATTR(NAME, ...) TEST_GENERIC_ATTR_N(*inst, InstAttrKind, NAME)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr0) {
  DECLS(os, *inst);
#define INST_ATTR_0(...) TEST_ATTR_0(os, *inst, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr1) {
  DECLS(os, *inst);
#define INST_ATTR_1(...) TEST_ATTR_1(os, *inst, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr2) {
  DECLS(os, *inst);
#define INST_ATTR_2(...) TEST_ATTR_2(os, *inst, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr3) {
  DECLS(os, *inst);
#define INST_ATTR_3(...) TEST_ATTR_3(os, *inst, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr4) {
  DECLS(os, *inst);
#define INST_ATTR_4(...) TEST_ATTR_4(os, *inst, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr5) {
  DECLS(os, *inst);
#define INST_ATTR_5(...) TEST_ATTR_5(os, *inst, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr6) {
  DECLS(os, *inst);
#define INST_ATTR_6(...) TEST_ATTR_6(os, *inst, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr7) {
  DECLS(os, *inst);
#define INST_ATTR_7(...) TEST_ATTR_7(os, *inst, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr8) {
  DECLS(os, *inst);
#define INST_ATTR_8(...) TEST_ATTR_8(os, *inst, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attrLoop) {
  DECLS_LOOP(os, *inst, loopF, loopG, lis)
#define INST_ATTR_LOOP(...)                                                    \
  TEST_ATTR_LOOP(os, *inst, loopF, loopG, lis, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

} // namespace
