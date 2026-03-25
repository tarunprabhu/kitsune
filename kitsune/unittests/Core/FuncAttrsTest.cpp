//===- FuncAttrsTest.cpp - Unit tests for Kitsune's function attributes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/FuncAttrs.h"
#include "TestAttrsCommon.h"
#include "TestValues.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

// The standard accessors do not allow us to create invalid attributes. To
// create one, we have to know how these are added to the function. This is not
// unreasonable since the create functions are a fairly thin wrappers around
// LLVM's existing support.
static void addMetadata(Function &f, StringRef name, ArrayRef<Metadata *> ops) {
  LLVMContext &ctx = f.getContext();
  MDNode *md = MDNode::get(ctx, ops);

  f.addMetadata(name, *md);
}

// Create metadata consisting of `n` "empty" operands.
static void addMetadata(Function &f, StringRef name, unsigned n) {
  LLVMContext &ctx = f.getContext();
  MDNode *mdEmpty = MDNode::get(ctx, {});
  SmallVector<Metadata *, 8> ops;

  ops.append(n, mdEmpty);
  addMetadata(f, name, ops);
}

TEST(KitFuncAttrs, attrName) {
#define FUNC_ATTR(NAME, IRNAME, TYPE)                                          \
  EXPECT_EQ(getAttrName(FuncAttrKind::NAME), IRNAME);                          \
  EXPECT_TRUE(getAttrName(FuncAttrKind::NAME).starts_with("kit.func."));
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attrKind) {
  EXPECT_EQ(getFuncAttrKind("keble"), std::nullopt);
#define FUNC_ATTR(NAME, IRNAME, TYPE)                                          \
  EXPECT_EQ(getFuncAttrKind(IRNAME), FuncAttrKind::NAME);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

#define DECLS(OS, OBJ)                                                         \
  std::string buf;                                                             \
  raw_string_ostream OS(buf);                                                  \
  LLVMContext ctx;                                                             \
  Type *voidTy = Type::getVoidTy(ctx);                                         \
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);       \
  [[maybe_unused]] Function OBJ =                                              \
      Function::Create(fty, GlobalValue::ExternalLinkage)

TEST(KitFuncAttrs, verifyGeneric) {
  DECLS(os, *f);
#define FUNC_ATTR_0(NAME, IRNAME)                                              \
  TEST_GENERIC_VERIFY_0(os, *f, FuncAttrKind, NAME, IRNAME);
#define FUNC_ATTR(NAME, IRNAME, TYPE)                                          \
  TEST_GENERIC_VERIFY_N(os, *f, FuncAttrKind, NAME, IRNAME);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attrsGeneric) {
  DECLS(os, *f);

#define FUNC_ATTR_0(NAME, IRNAME) TEST_GENERIC_ATTR_0(*f, FuncAttrKind, NAME);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_0(NAME, IRNAME)
#define FUNC_ATTR(NAME, IRNAME, TYPE)                                          \
  TEST_GENERIC_ATTR_N(*f, FuncAttrKind, NAME);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr0) {
  DECLS(os, *f);
#define FUNC_ATTR_0(NAME, IRNAME) TEST_ATTR_0(os, *f, NAME, IRNAME);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr1) {
  DECLS(os, *f);
#define FUNC_ATTR_1(NAME, IRNAME, TYPE) TEST_ATTR_1(os, *f, NAME, IRNAME, TYPE);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr2) {
  DECLS(os, *f);
#define FUNC_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)        \
  TEST_ATTR_2(os, *f, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr3) {
  DECLS(os, *f);
#define FUNC_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2)                                               \
  TEST_ATTR_3(os, *f, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
              ETY2, ENAME2, EN2);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr4) {
  DECLS(os, *f);
#define FUNC_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3)                            \
  TEST_ATTR_4(os, *f, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
              ETY2, ENAME2, EN2, ETY3, ENAME3, EN3);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr5) {
  DECLS(os, *f);
#define FUNC_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)         \
  TEST_ATTR_5(os, *f, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
              ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr6) {
  DECLS(os, *f);
#define FUNC_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5)                                               \
  TEST_ATTR_6(os, *f, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
              ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
              ENAME5, EN5);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr7) {
  DECLS(os, *f);
#define FUNC_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6)                            \
  TEST_ATTR_7(os, *f, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
              ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
              ENAME5, EN5, ETY6, ENAME6, EN6);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr8) {
  DECLS(os, *f);
#define FUNC_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)         \
  TEST_ATTR_8(os, *f, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1,      \
              ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
              ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

} // namespace
