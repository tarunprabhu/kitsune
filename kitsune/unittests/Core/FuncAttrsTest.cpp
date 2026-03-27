//===- FuncAttrsTest.cpp - Unit tests for Kitsune's function attributes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/FuncAttrs.h"
#include "TestAttrsCommon.h"
#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Core/FuncUtils.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

static void addMetadata(Function &f, StringRef attrName,
                        ArrayRef<Metadata *> attrVals) {
  LLVMContext &ctx = f.getContext();
  MDNode *attrList = getAttrList(f);
  MDNode *newAttrList = getAttrListWith(attrName, attrVals, attrList, ctx);

  f.setMetadata(LLVMContext::MD_kit_func_attrs, newAttrList);
}

// Create metadata consisting of `n` "empty" operands.
static void addMetadata(Function &f, StringRef attrName, unsigned n) {
  LLVMContext &ctx = f.getContext();
  MDNode *mdEmpty = MDNode::get(ctx, {});
  SmallVector<Metadata *, 8> attrVals;

  attrVals.append(n, mdEmpty);
  addMetadata(f, attrName, attrVals);
}

TEST(KitFuncAttrs, attrName) {
#define FUNC_ATTR(NAME, IRNAME, ...)                                           \
  EXPECT_EQ(getAttrName(FuncAttrKind::NAME), IRNAME);                          \
  EXPECT_TRUE(getAttrName(FuncAttrKind::NAME).starts_with("kit.func."));
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attrKind) {
  EXPECT_EQ(getFuncAttrKind("keble"), std::nullopt);
#define FUNC_ATTR(NAME, IRNAME, ...)                                           \
  EXPECT_EQ(getFuncAttrKind(IRNAME), FuncAttrKind::NAME);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

#define DECLS(OS, OBJ)                                                         \
  std::string buf;                                                             \
  raw_string_ostream OS(buf);                                                  \
  LLVMContext ctx;                                                             \
  Module m("", ctx);                                                           \
  Type *voidTy = Type::getVoidTy(ctx);                                         \
  FunctionType *fty = FunctionType::get(voidTy, {}, /*IsVarArg=*/false);       \
  [[maybe_unused]] Function OBJ =                                              \
      cast<Function>(m.getOrInsertFunction("f", fty).getCallee());

TEST(KitFuncAttrs, verifyGeneric) {
  DECLS(os, *f);
#define FUNC_ATTR_0(NAME, IRNAME, ...)                                         \
  TEST_GENERIC_VERIFY_0(os, *f, FuncAttrKind, NAME, IRNAME)
#define FUNC_ATTR(NAME, IRNAME, ...)                                           \
  TEST_GENERIC_VERIFY_N(os, *f, FuncAttrKind, NAME, IRNAME)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attrsGeneric) {
  DECLS(os, *f);

#define FUNC_ATTR_0(NAME, ...) TEST_GENERIC_ATTR_0(*f, FuncAttrKind, NAME)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_0(...)
#define FUNC_ATTR(NAME, ...) TEST_GENERIC_ATTR_N(*f, FuncAttrKind, NAME)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr0) {
  DECLS(os, *f);
#define FUNC_ATTR_0(...) TEST_ATTR_0(os, *f, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr1) {
  DECLS(os, *f);
#define FUNC_ATTR_1(...) TEST_ATTR_1(os, *f, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr2) {
  DECLS(os, *f);
#define FUNC_ATTR_2(...) TEST_ATTR_2(os, *f, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr3) {
  DECLS(os, *f);
#define FUNC_ATTR_3(...) TEST_ATTR_3(os, *f, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr4) {
  DECLS(os, *f);
#define FUNC_ATTR_4(...) TEST_ATTR_4(os, *f, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr5) {
  DECLS(os, *f);
#define FUNC_ATTR_5(...) TEST_ATTR_5(os, *f, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr6) {
  DECLS(os, *f);
#define FUNC_ATTR_6(...) TEST_ATTR_6(os, *f, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr7) {
  DECLS(os, *f);
#define FUNC_ATTR_7(...) TEST_ATTR_7(os, *f, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr8) {
  DECLS(os, *f);
#define FUNC_ATTR_8(...) TEST_ATTR_8(os, *f, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attrLoop) {
  DECLS_LOOP(os, *f, loopF, loopG, lis)
#define FUNC_ATTR_LOOP(...)                                                    \
  TEST_ATTR_LOOP(os, *f, loopF, loopG, lis, __VA_ARGS__)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

} // namespace
