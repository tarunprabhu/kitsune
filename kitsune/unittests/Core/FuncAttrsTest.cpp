//==- FuncAttrsTest.cpp - Unit tests for Kitsune's function attributes -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/FuncAttrs.h"
#include "llvm/IR/Function.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitFuncAttrs, attrName) {
#define FUNC_ATTR(NAME, TYPE, IRNAME)                                          \
  EXPECT_EQ(getAttrName(FuncAttrKind::NAME), IRNAME);                          \
  EXPECT_TRUE(getAttrName(FuncAttrKind::NAME).starts_with("kit.func."));
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attrKind) {
  EXPECT_EQ(getFuncAttrKind("whoops"), std::nullopt);

#define FUNC_ATTR(NAME, TYPE, IRNAME)                                          \
  EXPECT_EQ(getFuncAttrKind(IRNAME), FuncAttrKind::NAME);
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

#define CHECK_GENERIC_FLAG(NAME, FUNC)                                         \
  EXPECT_FALSE(hasAttr(FUNC, FuncAttrKind::NAME));                             \
  addAttr(FUNC, FuncAttrKind::NAME);                                           \
  EXPECT_TRUE(hasAttr(FUNC, FuncAttrKind::NAME));                              \
  removeAttr(FUNC, FuncAttrKind::NAME);                                        \
  EXPECT_FALSE(hasAttr(FUNC, FuncAttrKind::NAME));

#define CHECK_GENERIC(NAME, FUNC, VAL)                                         \
  EXPECT_FALSE(hasAttr(FUNC, FuncAttrKind::NAME));                             \
  add##NAME##Attr(FUNC, VAL);                                                  \
  EXPECT_TRUE(hasAttr(FUNC, FuncAttrKind::NAME));                              \
  removeAttr(FUNC, FuncAttrKind::NAME);                                        \
  EXPECT_FALSE(hasAttr(FUNC, FuncAttrKind::NAME));                             \
  EXPECT_EXIT(addAttr(FUNC, FuncAttrKind::NAME), ::testing::ExitedWithCode(1), \
              "error: cannot add attribute");

TEST(KitFuncAttrs, attrsGeneric) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);
  [[maybe_unused]] Function *f =
      Function::Create(fty, GlobalValue::ExternalLinkage);

#define FUNC_ATTR_FLAG(NAME, IRNAME) CHECK_GENERIC_FLAG(NAME, *f)
#define FUNC_ATTR_ENUM(NAME, IRNAME, TYPE) CHECK_GENERIC(NAME, *f, (TYPE)1)
#define FUNC_ATTR_F32(NAME, IRNAME) CHECK_GENERIC(NAME, *f, 3.14F)
#define FUNC_ATTR_F64(NAME, IRNAME) CHECK_GENERIC(NAME, *f, 3.141592653)
#define FUNC_ATTR_I32(NAME, IRNAME) CHECK_GENERIC(NAME, *f, 97)
#define FUNC_ATTR_I64(NAME, IRNAME) CHECK_GENERIC(NAME, *f, 53L)
#define FUNC_ATTR_STR(NAME, IRNAME) CHECK_GENERIC(NAME, *f, "strathclyde")
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, flagAttrs) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);
  [[maybe_unused]] Function *f =
      Function::Create(fty, GlobalValue::ExternalLinkage);

#define FUNC_ATTR_FLAG(NAME, IRNAME)                                           \
  EXPECT_FALSE(has##NAME##Attr(*f));                                           \
                                                                               \
  add##NAME##Attr(*f);                                                         \
  EXPECT_TRUE(has##NAME##Attr(*f));                                            \
                                                                               \
  add##NAME##Attr(*f);                                                         \
  EXPECT_TRUE(has##NAME##Attr(*f));                                            \
                                                                               \
  remove##NAME##Attr(*f);                                                      \
  EXPECT_FALSE(has##NAME##Attr(*f));

#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

#define CHECK_ACCESSORS(NAME, FUNC, VAL1, VAL2)                                \
  EXPECT_FALSE(has##NAME##Attr(FUNC));                                         \
                                                                               \
  add##NAME##Attr(FUNC, VAL1);                                                 \
  EXPECT_TRUE(has##NAME##Attr(FUNC));                                          \
  EXPECT_EQ(get##NAME##Attr(FUNC), (VAL1));                                    \
                                                                               \
  add##NAME##Attr(FUNC, VAL2);                                                 \
  EXPECT_TRUE(has##NAME##Attr(FUNC));                                          \
  EXPECT_EQ(get##NAME##Attr(FUNC), (VAL2));                                    \
                                                                               \
  remove##NAME##Attr(FUNC);                                                    \
  EXPECT_FALSE(has##NAME##Attr(FUNC));

TEST(KitFuncAttrs, enumAttrs) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);
  [[maybe_unused]] Function *f =
      Function::Create(fty, GlobalValue::ExternalLinkage);

  // WARNING: This is somewhat risky because there is no guarantee that the
  // integer values 1 and 2 will be valid for every enum type that we may have
  // an attribute for. If this ever happens, change this test. It may be
  // sufficient to just check for some enum-valued attribute instead of all of
  // them.
#define FUNC_ATTR_ENUM(NAME, IRNAME, TYPE)                                     \
  CHECK_ACCESSORS(NAME, *f, (TYPE)1, (TYPE)2)

#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, f32Test) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);
  [[maybe_unused]] Function *f =
      Function::Create(fty, GlobalValue::ExternalLinkage);

#define FUNC_ATTR_F32(NAME, IRNAME)                                            \
  CHECK_ACCESSORS(NAME, *f, 3.14159F, 2.71828F)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, f64Test) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);
  [[maybe_unused]] Function *f =
      Function::Create(fty, GlobalValue::ExternalLinkage);

#define FUNC_ATTR_F64(NAME, IRNAME)                                            \
  CHECK_ACCESSORS(NAME, *f, 3.1415926535, 2.71828)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, i32Test) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);
  [[maybe_unused]] Function *f =
      Function::Create(fty, GlobalValue::ExternalLinkage);

#define FUNC_ATTR_I32(NAME, IRNAME) CHECK_ACCESSORS(NAME, *f, 13, 19)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, i64Test) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);
  [[maybe_unused]] Function *f =
      Function::Create(fty, GlobalValue::ExternalLinkage);

#define FUNC_ATTR_I64(NAME, IRNAME) CHECK_ACCESSORS(NAME, *f, 101L, 103L)
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, strTest) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);
  [[maybe_unused]] Function *f =
      Function::Create(fty, GlobalValue::ExternalLinkage);

#define FUNC_ATTR_STR(NAME, IRNAME)                                            \
  CHECK_ACCESSORS(NAME, *f, "lucy cavendish", "wolfson")
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

} // namespace
