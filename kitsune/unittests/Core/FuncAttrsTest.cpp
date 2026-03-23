//==- FuncAttrsTest.cpp - Unit tests for Kitsune's function attributes -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/FuncAttrs.h"
#include "TestValues.h"
#include "llvm/IR/Function.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

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

TEST(KitFuncAttrs, attrsGeneric) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);
  [[maybe_unused]] Function *f =
      Function::Create(fty, GlobalValue::ExternalLinkage);

#define FUNC_ATTR_0(NAME, IRNAME)                                              \
  EXPECT_FALSE(hasAttr(*f, FuncAttrKind::NAME));                               \
  addAttr(*f, FuncAttrKind::NAME);                                             \
  EXPECT_TRUE(hasAttr(*f, FuncAttrKind::NAME));                                \
  addAttr(*f, FuncAttrKind::NAME);                                             \
  EXPECT_TRUE(hasAttr(*f, FuncAttrKind::NAME));                                \
  removeAttr(*f, FuncAttrKind::NAME);                                          \
  EXPECT_FALSE(hasAttr(*f, FuncAttrKind::NAME));
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"

#define FUNC_ATTR_0(NAME, IRNAME)
#define FUNC_ATTR(NAME, IRNAME, TYPE)                                          \
  EXPECT_EXIT(addAttr(*f, FuncAttrKind::NAME), ::testing::ExitedWithCode(1),   \
              "error: cannot add attribute");
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr0) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);
  [[maybe_unused]] Function *f =
      Function::Create(fty, GlobalValue::ExternalLinkage);

#define FUNC_ATTR_0(NAME, IRNAME)                                              \
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

TEST(KitFuncAttrs, attr1) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);
  [[maybe_unused]] Function *f =
      Function::Create(fty, GlobalValue::ExternalLinkage);

#define FUNC_ATTR_1(NAME, IRNAME, TYPE)                                        \
  EXPECT_FALSE(has##NAME##Attr(*f));                                           \
                                                                               \
  add##NAME##Attr(*f, get<TYPE>(0));                                           \
  EXPECT_TRUE(has##NAME##Attr(*f));                                            \
  EXPECT_EQ(get##NAME##Attr(*f), get<TYPE>(0));                                \
                                                                               \
  add##NAME##Attr(*f, get<TYPE>(1));                                           \
  EXPECT_TRUE(has##NAME##Attr(*f));                                            \
  EXPECT_EQ(get##NAME##Attr(*f), get<TYPE>(1));                                \
                                                                               \
  remove##NAME##Attr(*f);                                                      \
  EXPECT_FALSE(has##NAME##Attr(*f));
#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr2) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);
  [[maybe_unused]] Function *f =
      Function::Create(fty, GlobalValue::ExternalLinkage);

#define FUNC_ATTR_2(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1)        \
  EXPECT_FALSE(has##NAME##Attr(*f));                                           \
                                                                               \
  add##NAME##Attr(*f, get<ETY0>(0), get<ETY1>(1));                             \
  EXPECT_TRUE(has##NAME##Attr(*f));                                            \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*f), get<ETY0>(0));                  \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*f), get<ETY1>(1));                  \
                                                                               \
  add##NAME##Attr(*f, get<ETY0>(1), get<ETY1>(0));                             \
  EXPECT_TRUE(has##NAME##Attr(*f));                                            \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*f), get<ETY0>(1));                  \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*f), get<ETY1>(0));                  \
                                                                               \
  remove##NAME##Attr(*f);                                                      \
  EXPECT_FALSE(has##NAME##Attr(*f));

#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr3) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);
  [[maybe_unused]] Function *f =
      Function::Create(fty, GlobalValue::ExternalLinkage);

#define FUNC_ATTR_3(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2)                                               \
  EXPECT_FALSE(has##NAME##Attr(*f));                                           \
                                                                               \
  add##NAME##Attr(*f, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2));               \
  EXPECT_TRUE(has##NAME##Attr(*f));                                            \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*f), get<ETY0>(0));                  \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*f), get<ETY1>(1));                  \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*f), get<ETY2>(2));                  \
                                                                               \
  add##NAME##Attr(*f, get<ETY0>(2), get<ETY1>(1), get<ETY2>(0));               \
  EXPECT_TRUE(has##NAME##Attr(*f));                                            \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*f), get<ETY0>(2));                  \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*f), get<ETY1>(1));                  \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*f), get<ETY2>(0));                  \
                                                                               \
  remove##NAME##Attr(*f);                                                      \
  EXPECT_FALSE(has##NAME##Attr(*f));

#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr4) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);
  [[maybe_unused]] Function *f =
      Function::Create(fty, GlobalValue::ExternalLinkage);

#define FUNC_ATTR_4(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3)                            \
  EXPECT_FALSE(has##NAME##Attr(*f));                                           \
                                                                               \
  add##NAME##Attr(*f, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2), get<ETY3>(3)); \
  EXPECT_TRUE(has##NAME##Attr(*f));                                            \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*f), get<ETY0>(0));                  \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*f), get<ETY1>(1));                  \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*f), get<ETY2>(2));                  \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*f), get<ETY3>(3));                  \
                                                                               \
  add##NAME##Attr(*f, get<ETY0>(3), get<ETY1>(2), get<ETY2>(1), get<ETY3>(0)); \
  EXPECT_TRUE(has##NAME##Attr(*f));                                            \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*f), get<ETY0>(3));                  \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*f), get<ETY1>(2));                  \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*f), get<ETY2>(1));                  \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*f), get<ETY3>(0));                  \
                                                                               \
  remove##NAME##Attr(*f);                                                      \
  EXPECT_FALSE(has##NAME##Attr(*f));

#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr5) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);
  [[maybe_unused]] Function *f =
      Function::Create(fty, GlobalValue::ExternalLinkage);

#define FUNC_ATTR_5(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4)         \
  EXPECT_FALSE(has##NAME##Attr(*f));                                           \
                                                                               \
  add##NAME##Attr(*f, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2), get<ETY3>(3),  \
                  get<ETY4>(4));                                               \
  EXPECT_TRUE(has##NAME##Attr(*f));                                            \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*f), get<ETY0>(0));                  \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*f), get<ETY1>(1));                  \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*f), get<ETY2>(2));                  \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*f), get<ETY3>(3));                  \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*f), get<ETY4>(4));                  \
                                                                               \
  add##NAME##Attr(*f, get<ETY0>(4), get<ETY1>(3), get<ETY2>(2), get<ETY3>(1),  \
                  get<ETY4>(0));                                               \
  EXPECT_TRUE(has##NAME##Attr(*f));                                            \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*f), get<ETY0>(4));                  \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*f), get<ETY1>(3));                  \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*f), get<ETY2>(2));                  \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*f), get<ETY3>(1));                  \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*f), get<ETY4>(0));                  \
                                                                               \
  remove##NAME##Attr(*f);                                                      \
  EXPECT_FALSE(has##NAME##Attr(*f));

#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr6) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);
  [[maybe_unused]] Function *f =
      Function::Create(fty, GlobalValue::ExternalLinkage);

#define FUNC_ATTR_6(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5)                                               \
  EXPECT_FALSE(has##NAME##Attr(*f));                                           \
                                                                               \
  add##NAME##Attr(*f, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2), get<ETY3>(3),  \
                  get<ETY4>(4), get<ETY5>(5));                                 \
  EXPECT_TRUE(has##NAME##Attr(*f));                                            \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*f), get<ETY0>(0));                  \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*f), get<ETY1>(1));                  \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*f), get<ETY2>(2));                  \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*f), get<ETY3>(3));                  \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*f), get<ETY4>(4));                  \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(*f), get<ETY5>(5));                  \
                                                                               \
  add##NAME##Attr(*f, get<ETY0>(5), get<ETY1>(4), get<ETY2>(3), get<ETY3>(2),  \
                  get<ETY4>(1), get<ETY5>(0));                                 \
  EXPECT_TRUE(has##NAME##Attr(*f));                                            \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*f), get<ETY0>(5));                  \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*f), get<ETY1>(4));                  \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*f), get<ETY2>(3));                  \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*f), get<ETY3>(2));                  \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*f), get<ETY4>(1));                  \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(*f), get<ETY5>(0));                  \
                                                                               \
  remove##NAME##Attr(*f);                                                      \
  EXPECT_FALSE(has##NAME##Attr(*f));

#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr7) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);
  [[maybe_unused]] Function *f =
      Function::Create(fty, GlobalValue::ExternalLinkage);

#define FUNC_ATTR_7(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6)                            \
  EXPECT_FALSE(has##NAME##Attr(*f));                                           \
                                                                               \
  add##NAME##Attr(*f, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2), get<ETY3>(3),  \
                  get<ETY4>(4), get<ETY5>(5), get<ETY6>(6));                   \
  EXPECT_TRUE(has##NAME##Attr(*f));                                            \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*f), get<ETY0>(0));                  \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*f), get<ETY1>(1));                  \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*f), get<ETY2>(2));                  \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*f), get<ETY3>(3));                  \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*f), get<ETY4>(4));                  \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(*f), get<ETY5>(5));                  \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(*f), get<ETY6>(6));                  \
                                                                               \
  add##NAME##Attr(*f, get<ETY0>(6), get<ETY1>(5), get<ETY2>(4), get<ETY3>(3),  \
                  get<ETY4>(2), get<ETY5>(1), get<ETY6>(0));                   \
  EXPECT_TRUE(has##NAME##Attr(*f));                                            \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*f), get<ETY0>(6));                  \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*f), get<ETY1>(5));                  \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*f), get<ETY2>(4));                  \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*f), get<ETY3>(3));                  \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*f), get<ETY4>(2));                  \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(*f), get<ETY5>(1));                  \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(*f), get<ETY6>(0));                  \
                                                                               \
  remove##NAME##Attr(*f);                                                      \
  EXPECT_FALSE(has##NAME##Attr(*f));

#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

TEST(KitFuncAttrs, attr8) {
  LLVMContext ctx;
  Type *voidTy = Type::getVoidTy(ctx);
  FunctionType *fty = FunctionType::get(voidTy, {}, /*isVarArg=*/false);
  [[maybe_unused]] Function *f =
      Function::Create(fty, GlobalValue::ExternalLinkage);

#define FUNC_ATTR_8(NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1, EN1, ETY2,  \
                    ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4, EN4, ETY5,   \
                    ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7, EN7)         \
  EXPECT_FALSE(has##NAME##Attr(*f));                                           \
                                                                               \
  add##NAME##Attr(*f, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2), get<ETY3>(3),  \
                  get<ETY4>(4), get<ETY5>(5), get<ETY6>(6), get<ETY7>(7));     \
  EXPECT_TRUE(has##NAME##Attr(*f));                                            \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*f), get<ETY0>(0));                  \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*f), get<ETY1>(1));                  \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*f), get<ETY2>(2));                  \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*f), get<ETY3>(3));                  \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*f), get<ETY4>(4));                  \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(*f), get<ETY5>(5));                  \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(*f), get<ETY6>(6));                  \
  EXPECT_EQ(get##ENAME7##From##NAME##Attr(*f), get<ETY7>(7));                  \
                                                                               \
  add##NAME##Attr(*f, get<ETY0>(7), get<ETY1>(6), get<ETY2>(5), get<ETY3>(4),  \
                  get<ETY4>(3), get<ETY5>(2), get<ETY6>(1), get<ETY7>(0));     \
  EXPECT_TRUE(has##NAME##Attr(*f));                                            \
  EXPECT_EQ(get##ENAME0##From##NAME##Attr(*f), get<ETY0>(7));                  \
  EXPECT_EQ(get##ENAME1##From##NAME##Attr(*f), get<ETY1>(6));                  \
  EXPECT_EQ(get##ENAME2##From##NAME##Attr(*f), get<ETY2>(5));                  \
  EXPECT_EQ(get##ENAME3##From##NAME##Attr(*f), get<ETY3>(4));                  \
  EXPECT_EQ(get##ENAME4##From##NAME##Attr(*f), get<ETY4>(3));                  \
  EXPECT_EQ(get##ENAME5##From##NAME##Attr(*f), get<ETY5>(2));                  \
  EXPECT_EQ(get##ENAME6##From##NAME##Attr(*f), get<ETY6>(1));                  \
  EXPECT_EQ(get##ENAME7##From##NAME##Attr(*f), get<ETY7>(0));                  \
                                                                               \
  remove##NAME##Attr(*f);                                                      \
  EXPECT_FALSE(has##NAME##Attr(*f));

#define GET_FUNC_ATTRS
#include "kitsune/Core/FuncAttrs.inc"
}

} // namespace
