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
#include "kitsune/Core/Verifier.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

template <typename T, InstAttrKind Attr> static T get(unsigned idx) {
  return ::get_<T>(idx);
}

// In some cases, it is difficult to construct a valid attribute - for instance
// if the attribute initializer must be valid bitcode. In such cases, we test
// everything but the verifier. lit tests must be added to ensure that the
// verification works correctly.
static constexpr bool verifyAttr(InstAttrKind attr) { return true; }

static void addMetadata(Instruction &inst, StringRef attrName,
                        ArrayRef<Metadata *> attrVals) {
  LLVMContext &ctx = inst.getContext();
  MDNode *attrList = getRawAttrList(inst);
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

#define DECLS(OBJ)                                                             \
  std::string buf;                                                             \
  raw_string_ostream OS(buf);                                                  \
  [[maybe_unused]] KitVerifier VOS(&OS);                                       \
  [[maybe_unused]] KitVerifier VNULL;                                          \
  LLVMContext ctx;                                                             \
  Type *voidTy = Type::getVoidTy(ctx);                                         \
  FunctionType *fty = FunctionType::get(voidTy, {}, /*IsVarArg=*/false);       \
  Module m("", ctx);                                                           \
  Function *f = cast<Function>(m.getOrInsertFunction("f", fty).getCallee());   \
  BasicBlock *bb = BasicBlock::Create(ctx, "entry", f);                        \
  [[maybe_unused]] ReturnInst OBJ =                                            \
      ReturnInst::Create(ctx, /*retVal=*/nullptr, bb->getFirstInsertionPt());

TEST(KitInstAttrs, verifyGeneric) {
  DECLS(*inst);
#define INST_ATTR_0(NAME, IRNAME, ...)                                         \
  TEST_GENERIC_VERIFY_0(*inst, InstAttrKind, NAME, IRNAME)
#define INST_ATTR(NAME, IRNAME, ...)                                           \
  TEST_GENERIC_VERIFY_N(*inst, InstAttrKind, NAME, IRNAME)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attrsGeneric) {
  DECLS(*inst);

#define INST_ATTR_0(NAME, ...) TEST_GENERIC_ATTR_0(*inst, InstAttrKind, NAME)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"

#define INST_ATTR_0(...)
#define INST_ATTR(NAME, ...) TEST_GENERIC_ATTR_N(*inst, InstAttrKind, NAME)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr0) {
  DECLS(*inst);
#define INST_ATTR_0(...) TEST_ATTR_0(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr1) {
  DECLS(*inst);
#define INST_ATTR_1(...) TEST_ATTR_1(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr2) {
  DECLS(*inst);
#define INST_ATTR_2(...) TEST_ATTR_2(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr3) {
  DECLS(*inst);
#define INST_ATTR_3(...) TEST_ATTR_3(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr4) {
  DECLS(*inst);
#define INST_ATTR_4(...) TEST_ATTR_4(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr5) {
  DECLS(*inst);
#define INST_ATTR_5(...) TEST_ATTR_5(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr6) {
  DECLS(*inst);
#define INST_ATTR_6(...) TEST_ATTR_6(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr7) {
  DECLS(*inst);
#define INST_ATTR_7(...) TEST_ATTR_7(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attr8) {
  DECLS(*inst);
#define INST_ATTR_8(...) TEST_ATTR_8(*inst, InstAttrKind, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attrLoop) {
  DECLS_LOOP(*inst, loopF, loopG, lis);
#define INST_ATTR_LOOP(...)                                                    \
  TEST_ATTR_LOOP(*inst, loopF, loopG, lis, __VA_ARGS__)
#define GET_INST_ATTRS
#include "kitsune/Core/InstAttrs.inc"
}

TEST(KitInstAttrs, attrRange) {
  DECLS(*inst);
  TEST_ATTR_ATTRS(*inst)
}

} // namespace
