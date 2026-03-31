//===- TestAttrsCommon.h - Utilities shared by tests for attrs -*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_UNITTEST_TEST_ATTRS_COMMON_H
#define KITSUNE_UNITTEST_TEST_ATTRS_COMMON_H

#include "TestUtils.h"
#include "TestValues.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"

// Add an attribute with the given name and of `n` "empty" values.
template <typename IRElem>
static void addAttr(IRElem &ir, llvm::StringRef attrName, unsigned n) {
  llvm::LLVMContext &ctx = getContext(ir);
  llvm::MDNode *mdEmpty = llvm::MDNode::get(ctx, {});
  llvm::SmallVector<llvm::Metadata *, 8> attrVals;

  attrVals.append(n, mdEmpty);
  llvm::detail::addAttr(ir, attrName, attrVals);
}

#define TEST_VERIFY_TRUE(OBJ, NAME)                                            \
  do {                                                                         \
    KitVerifier verifier(nullptr);                                             \
                                                                               \
    verify##NAME##Attr(verifier, OBJ);                                         \
    EXPECT_TRUE(verifier.result());                                            \
  } while (0)

#define TEST_VERIFY_FALSE(OBJ, NAME)                                           \
  do {                                                                         \
    KitVerifier verifier(nullptr);                                             \
                                                                               \
    verify##NAME##Attr(verifier, OBJ);                                         \
    EXPECT_FALSE(verifier.result());                                           \
  } while (0)

#define TEST_VERIFY_ERR_0(OBJ, NAME)                                           \
  do {                                                                         \
    std::string buf;                                                           \
    raw_string_ostream os(buf);                                                \
    KitVerifier verifier(&os);                                                 \
                                                                               \
    verify##NAME##Attr(verifier, OBJ);                                         \
    EXPECT_TRUE(verifier.result());                                            \
    EXPECT_TRUE(buf.empty());                                                  \
  } while (0)

#define TEST_VERIFY_ERR(OBJ, NAME, ERR)                                        \
  do {                                                                         \
    std::string buf;                                                           \
    raw_string_ostream os(buf);                                                \
    KitVerifier verifier(&os);                                                 \
                                                                               \
    verify##NAME##Attr(verifier, OBJ);                                         \
    EXPECT_FALSE(verifier.result());                                           \
    EXPECT_TRUE(StringRef(buf).contains(ERR));                                 \
  } while (0)

#define TEST_GENERIC_VERIFY_N(OBJ, KIND, NAME, IRNAME)                         \
  detail::addAttr(OBJ, IRNAME, {});                                            \
                                                                               \
  TEST_VERIFY_FALSE(OBJ, NAME);                                                \
  TEST_VERIFY_ERR(OBJ, NAME, "incorrect value count");                         \
                                                                               \
  remove##NAME##Attr((OBJ));

#define TEST_GENERIC_VERIFY_0(OBJ, KIND, NAME, IRNAME)                         \
  detail::addAttr(OBJ, IRNAME, MDString::get(ctx, ""));                        \
                                                                               \
  TEST_VERIFY_FALSE(OBJ, NAME);                                                \
  TEST_VERIFY_ERR(OBJ, NAME, "incorrect value count '1'");                     \
                                                                               \
  remove##NAME##Attr(OBJ);

#define TEST_GENERIC_ATTR_0(OBJ, KIND, NAME)                                   \
  EXPECT_FALSE(hasAttr(OBJ, KIND::NAME));                                      \
                                                                               \
  addAttr(OBJ, KIND::NAME);                                                    \
  EXPECT_TRUE(hasAttr(OBJ, KIND::NAME));                                       \
                                                                               \
  addAttr(OBJ, KIND::NAME);                                                    \
  EXPECT_TRUE(hasAttr(OBJ, KIND::NAME));                                       \
                                                                               \
  removeAttr(OBJ, KIND::NAME);                                                 \
  EXPECT_FALSE(hasAttr(OBJ, KIND::NAME));

#define TEST_GENERIC_ATTR_N(OBJ, KIND, NAME)                                   \
  EXPECT_EXIT(addAttr(OBJ, KIND::NAME), ::testing::ExitedWithCode(1),          \
              "error: cannot add attribute");

#define TEST_ATTR_0(OBJ, KIND, NAME, IRNAME, CUSTOMVERIFY)                     \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  add##NAME##Attr(OBJ);                                                        \
  EXPECT_TRUE(has##NAME##Attr(OBJ));                                           \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  add##NAME##Attr(OBJ);                                                        \
  EXPECT_TRUE(has##NAME##Attr(OBJ));                                           \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  remove##NAME##Attr(OBJ);                                                     \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  ::addAttr(OBJ, IRNAME, 0);                                                   \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
  TEST_VERIFY_ERR_0(OBJ, NAME);                                                \
  remove##NAME##Attr(OBJ);

#define TEST_ATTR_1(OBJ, KIND, NAME, IRNAME, CUSTOMVERIFY, TYPE)               \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  {                                                                            \
    TYPE v = get<TYPE, KIND::NAME>(0);                                         \
                                                                               \
    add##NAME##Attr(OBJ, v);                                                   \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    if constexpr (::verifyAttr(KIND::NAME))                                    \
      TEST_VERIFY_TRUE(OBJ, NAME);                                             \
    EXPECT_EQ(get##NAME##Attr(OBJ), v);                                        \
  }                                                                            \
                                                                               \
  {                                                                            \
    TYPE v = get<TYPE, KIND::NAME>(1);                                         \
                                                                               \
    add##NAME##Attr(OBJ, v);                                                   \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    if constexpr (::verifyAttr(KIND::NAME))                                    \
      TEST_VERIFY_TRUE(OBJ, NAME);                                             \
    EXPECT_EQ(get##NAME##Attr(OBJ), v);                                        \
  }                                                                            \
                                                                               \
  remove##NAME##Attr(OBJ);                                                     \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  ::addAttr(OBJ, IRNAME, 1);                                                   \
  TEST_VERIFY_FALSE(OBJ, NAME);                                                \
  TEST_VERIFY_ERR(OBJ, NAME, "missing value of type");                         \
  remove##NAME##Attr(OBJ);

#define TEST_ATTR_2(OBJ, KIND, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,  \
                    ETY1, ENAME1, EN1)                                         \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  {                                                                            \
    ETY0 v0 = get<ETY0, KIND::NAME>(0);                                        \
    ETY1 v1 = get<ETY1, KIND::NAME>(1);                                        \
                                                                               \
    add##NAME##Attr(OBJ, v0, v1);                                              \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    if constexpr (::verifyAttr(KIND::NAME))                                    \
      TEST_VERIFY_TRUE(OBJ, NAME);                                             \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), v0);                         \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), v1);                         \
  }                                                                            \
                                                                               \
  {                                                                            \
    ETY0 v0 = get<ETY0, KIND::NAME>(1);                                        \
    ETY1 v1 = get<ETY1, KIND::NAME>(0);                                        \
                                                                               \
    add##NAME##Attr(OBJ, v0, v1);                                              \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    if constexpr (::verifyAttr(KIND::NAME))                                    \
      TEST_VERIFY_TRUE(OBJ, NAME);                                             \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), v0);                         \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), v1);                         \
  }                                                                            \
                                                                               \
  remove##NAME##Attr(OBJ);                                                     \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  ::addAttr(OBJ, IRNAME, 2);                                                   \
  TEST_VERIFY_FALSE(OBJ, NAME);                                                \
  TEST_VERIFY_ERR(OBJ, NAME, "missing value of type");                         \
  remove##NAME##Attr(OBJ);

#define TEST_ATTR_3(OBJ, KIND, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,  \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2)                      \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  {                                                                            \
    ETY0 v0 = get<ETY0, KIND::NAME>(0);                                        \
    ETY1 v1 = get<ETY1, KIND::NAME>(1);                                        \
    ETY2 v2 = get<ETY2, KIND::NAME>(2);                                        \
                                                                               \
    add##NAME##Attr(OBJ, v0, v1, v2);                                          \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    if constexpr (::verifyAttr(KIND::NAME))                                    \
      TEST_VERIFY_TRUE(OBJ, NAME);                                             \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), v0);                         \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), v1);                         \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), v2);                         \
  }                                                                            \
                                                                               \
  {                                                                            \
    ETY0 v0 = get<ETY0, KIND::NAME>(2);                                        \
    ETY1 v1 = get<ETY1, KIND::NAME>(1);                                        \
    ETY2 v2 = get<ETY2, KIND::NAME>(0);                                        \
                                                                               \
    add##NAME##Attr(OBJ, v0, v1, v2);                                          \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    if constexpr (::verifyAttr(KIND::NAME))                                    \
      TEST_VERIFY_TRUE(OBJ, NAME);                                             \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), v0);                         \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), v1);                         \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), v2);                         \
  }                                                                            \
                                                                               \
  remove##NAME##Attr(OBJ);                                                     \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  ::addAttr(OBJ, IRNAME, 3);                                                   \
  TEST_VERIFY_FALSE(OBJ, NAME);                                                \
  TEST_VERIFY_ERR(OBJ, NAME, "missing value of type");                         \
  remove##NAME##Attr(OBJ);

#define TEST_ATTR_4(OBJ, KIND, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,  \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3)   \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  {                                                                            \
    ETY0 v0 = get<ETY0, KIND::NAME>(0);                                        \
    ETY1 v1 = get<ETY1, KIND::NAME>(1);                                        \
    ETY2 v2 = get<ETY2, KIND::NAME>(2);                                        \
    ETY3 v3 = get<ETY3, KIND::NAME>(3);                                        \
                                                                               \
    add##NAME##Attr(OBJ, v0, v1, v2, v3);                                      \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    if constexpr (::verifyAttr(KIND::NAME))                                    \
      TEST_VERIFY_TRUE(OBJ, NAME);                                             \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), v0);                         \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), v1);                         \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), v2);                         \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), v3);                         \
  }                                                                            \
                                                                               \
  {                                                                            \
    ETY0 v0 = get<ETY0, KIND::NAME>(3);                                        \
    ETY1 v1 = get<ETY1, KIND::NAME>(2);                                        \
    ETY2 v2 = get<ETY2, KIND::NAME>(1);                                        \
    ETY3 v3 = get<ETY3, KIND::NAME>(0);                                        \
                                                                               \
    add##NAME##Attr(OBJ, v0, v1, v2, v3);                                      \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    if constexpr (::verifyAttr(KIND::NAME))                                    \
      TEST_VERIFY_TRUE(OBJ, NAME);                                             \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), v0);                         \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), v1);                         \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), v2);                         \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), v3);                         \
  }                                                                            \
                                                                               \
  remove##NAME##Attr(OBJ);                                                     \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  ::addAttr(OBJ, IRNAME, 4);                                                   \
  TEST_VERIFY_FALSE(OBJ, NAME);                                                \
  TEST_VERIFY_ERR(OBJ, NAME, "missing value of type");                         \
  remove##NAME##Attr(OBJ);

#define TEST_ATTR_5(OBJ, KIND, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,  \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4)                                         \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  {                                                                            \
    ETY0 v0 = get<ETY0, KIND::NAME>(0);                                        \
    ETY1 v1 = get<ETY1, KIND::NAME>(1);                                        \
    ETY2 v2 = get<ETY2, KIND::NAME>(2);                                        \
    ETY3 v3 = get<ETY3, KIND::NAME>(3);                                        \
    ETY4 v4 = get<ETY4, KIND::NAME>(4);                                        \
                                                                               \
    add##NAME##Attr(OBJ, v0, v1, v2, v3, v4);                                  \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    if constexpr (::verifyAttr(KIND::NAME))                                    \
      TEST_VERIFY_TRUE(OBJ, NAME);                                             \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), v0);                         \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), v1);                         \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), v2);                         \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), v3);                         \
    EXPECT_EQ(get##ENAME4##From##NAME##Attr(OBJ), v4);                         \
  }                                                                            \
                                                                               \
  {                                                                            \
    ETY0 v0 = get<ETY0, KIND::NAME>(4);                                        \
    ETY1 v1 = get<ETY1, KIND::NAME>(3);                                        \
    ETY2 v2 = get<ETY2, KIND::NAME>(2);                                        \
    ETY3 v3 = get<ETY3, KIND::NAME>(1);                                        \
    ETY4 v4 = get<ETY4, KIND::NAME>(0);                                        \
                                                                               \
    add##NAME##Attr(OBJ, v0, v1, v2, v3, v4);                                  \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    if constexpr (::verifyAttr(KIND::NAME))                                    \
      TEST_VERIFY_TRUE(OBJ, NAME);                                             \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), v0);                         \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), v1);                         \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), v2);                         \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), v3);                         \
    EXPECT_EQ(get##ENAME4##From##NAME##Attr(OBJ), v4);                         \
  }                                                                            \
                                                                               \
  remove##NAME##Attr(OBJ);                                                     \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  ::addAttr(OBJ, IRNAME, 5);                                                   \
  TEST_VERIFY_FALSE(OBJ, NAME);                                                \
  TEST_VERIFY_ERR(OBJ, NAME, "missing value of type");                         \
  remove##NAME##Attr(OBJ);

#define TEST_ATTR_6(OBJ, KIND, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,  \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4, ETY5, ENAME5, EN5)                      \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  {                                                                            \
    ETY0 v0 = get<ETY0, KIND::NAME>(0);                                        \
    ETY1 v1 = get<ETY1, KIND::NAME>(1);                                        \
    ETY2 v2 = get<ETY2, KIND::NAME>(2);                                        \
    ETY3 v3 = get<ETY3, KIND::NAME>(3);                                        \
    ETY4 v4 = get<ETY4, KIND::NAME>(4);                                        \
    ETY5 v5 = get<ETY5, KIND::NAME>(5);                                        \
                                                                               \
    add##NAME##Attr(OBJ, v0, v1, v2, v3, v4, v5);                              \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    if constexpr (::verifyAttr(KIND::NAME))                                    \
      TEST_VERIFY_TRUE(OBJ, NAME);                                             \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), v0);                         \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), v1);                         \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), v2);                         \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), v3);                         \
    EXPECT_EQ(get##ENAME4##From##NAME##Attr(OBJ), v4);                         \
    EXPECT_EQ(get##ENAME5##From##NAME##Attr(OBJ), v5);                         \
  }                                                                            \
                                                                               \
  {                                                                            \
    ETY0 v0 = get<ETY0, KIND::NAME>(5);                                        \
    ETY1 v1 = get<ETY1, KIND::NAME>(4);                                        \
    ETY2 v2 = get<ETY2, KIND::NAME>(3);                                        \
    ETY3 v3 = get<ETY3, KIND::NAME>(2);                                        \
    ETY4 v4 = get<ETY4, KIND::NAME>(1);                                        \
    ETY5 v5 = get<ETY5, KIND::NAME>(0);                                        \
                                                                               \
    add##NAME##Attr(OBJ, v0, v1, v2, v3, v4, v5);                              \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    if constexpr (::verifyAttr(KIND::NAME))                                    \
      TEST_VERIFY_TRUE(OBJ, NAME);                                             \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), v0);                         \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), v1);                         \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), v2);                         \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), v3);                         \
    EXPECT_EQ(get##ENAME4##From##NAME##Attr(OBJ), v4);                         \
    EXPECT_EQ(get##ENAME5##From##NAME##Attr(OBJ), v5);                         \
  }                                                                            \
                                                                               \
  remove##NAME##Attr(OBJ);                                                     \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  ::addAttr(OBJ, IRNAME, 6);                                                   \
  TEST_VERIFY_FALSE(OBJ, NAME);                                                \
  TEST_VERIFY_ERR(OBJ, NAME, "missing value of type");                         \
  remove##NAME##Attr(OBJ);

#define TEST_ATTR_7(OBJ, KIND, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,  \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4, ETY5, ENAME5, EN5, ETY6, ENAME6, EN6)   \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  {                                                                            \
    ETY0 v0 = get<ETY0, KIND::NAME>(0);                                        \
    ETY1 v1 = get<ETY1, KIND::NAME>(1);                                        \
    ETY2 v2 = get<ETY2, KIND::NAME>(2);                                        \
    ETY3 v3 = get<ETY3, KIND::NAME>(3);                                        \
    ETY4 v4 = get<ETY4, KIND::NAME>(4);                                        \
    ETY5 v5 = get<ETY5, KIND::NAME>(5);                                        \
    ETY6 v6 = get<ETY6, KIND::NAME>(6);                                        \
                                                                               \
    add##NAME##Attr(OBJ, v0, v1, v2, v3, v4, v5, v6);                          \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    if constexpr (::verifyAttr(KIND::NAME))                                    \
      TEST_VERIFY_TRUE(OBJ, NAME);                                             \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), v0);                         \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), v1);                         \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), v2);                         \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), v3);                         \
    EXPECT_EQ(get##ENAME4##From##NAME##Attr(OBJ), v4);                         \
    EXPECT_EQ(get##ENAME5##From##NAME##Attr(OBJ), v5);                         \
    EXPECT_EQ(get##ENAME6##From##NAME##Attr(OBJ), v6);                         \
  }                                                                            \
                                                                               \
  {                                                                            \
    ETY0 v0 = get<ETY0, KIND::NAME>(6);                                        \
    ETY1 v1 = get<ETY1, KIND::NAME>(5);                                        \
    ETY2 v2 = get<ETY2, KIND::NAME>(4);                                        \
    ETY3 v3 = get<ETY3, KIND::NAME>(3);                                        \
    ETY4 v4 = get<ETY4, KIND::NAME>(2);                                        \
    ETY5 v5 = get<ETY5, KIND::NAME>(1);                                        \
    ETY6 v6 = get<ETY6, KIND::NAME>(0);                                        \
                                                                               \
    add##NAME##Attr(OBJ, v0, v1, v2, v3, v4, v5, v6);                          \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    if constexpr (::verifyAttr(KIND::NAME))                                    \
      TEST_VERIFY_TRUE(OBJ, NAME);                                             \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), v0);                         \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), v1);                         \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), v2);                         \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), v3);                         \
    EXPECT_EQ(get##ENAME4##From##NAME##Attr(OBJ), v4);                         \
    EXPECT_EQ(get##ENAME5##From##NAME##Attr(OBJ), v5);                         \
    EXPECT_EQ(get##ENAME6##From##NAME##Attr(OBJ), v6);                         \
  }                                                                            \
                                                                               \
  remove##NAME##Attr(OBJ);                                                     \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  ::addAttr(OBJ, IRNAME, 7);                                                   \
  TEST_VERIFY_FALSE(OBJ, NAME);                                                \
  TEST_VERIFY_ERR(OBJ, NAME, "missing value of type");                         \
  remove##NAME##Attr(OBJ);

#define TEST_ATTR_8(OBJ, KIND, NAME, IRNAME, CUSTOMVERIFY, ETY0, ENAME0, EN0,  \
                    ETY1, ENAME1, EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3,   \
                    ETY4, ENAME4, EN4, ETY5, ENAME5, EN5, ETY6, ENAME6, EN6,   \
                    ETY7, ENAME7, EN7)                                         \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  {                                                                            \
    ETY0 v0 = get<ETY0, KIND::NAME>(0);                                        \
    ETY1 v1 = get<ETY1, KIND::NAME>(1);                                        \
    ETY2 v2 = get<ETY2, KIND::NAME>(2);                                        \
    ETY3 v3 = get<ETY3, KIND::NAME>(3);                                        \
    ETY4 v4 = get<ETY4, KIND::NAME>(4);                                        \
    ETY5 v5 = get<ETY5, KIND::NAME>(5);                                        \
    ETY6 v6 = get<ETY6, KIND::NAME>(6);                                        \
    ETY7 v7 = get<ETY7, KIND::NAME>(7);                                        \
                                                                               \
    add##NAME##Attr(OBJ, v0, v1, v2, v3, v4, v5, v6, v7);                      \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    if constexpr (::verifyAttr(KIND::NAME))                                    \
      TEST_VERIFY_TRUE(OBJ, NAME);                                             \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), v0);                         \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), v1);                         \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), v2);                         \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), v3);                         \
    EXPECT_EQ(get##ENAME4##From##NAME##Attr(OBJ), v4);                         \
    EXPECT_EQ(get##ENAME5##From##NAME##Attr(OBJ), v5);                         \
    EXPECT_EQ(get##ENAME6##From##NAME##Attr(OBJ), v6);                         \
    EXPECT_EQ(get##ENAME7##From##NAME##Attr(OBJ), v7);                         \
  }                                                                            \
                                                                               \
  {                                                                            \
    ETY0 v0 = get<ETY0, KIND::NAME>(7);                                        \
    ETY1 v1 = get<ETY1, KIND::NAME>(6);                                        \
    ETY2 v2 = get<ETY2, KIND::NAME>(5);                                        \
    ETY3 v3 = get<ETY3, KIND::NAME>(4);                                        \
    ETY4 v4 = get<ETY4, KIND::NAME>(3);                                        \
    ETY5 v5 = get<ETY5, KIND::NAME>(2);                                        \
    ETY6 v6 = get<ETY6, KIND::NAME>(1);                                        \
    ETY7 v7 = get<ETY7, KIND::NAME>(0);                                        \
                                                                               \
    add##NAME##Attr(OBJ, v0, v1, v2, v3, v4, v5, v6, v7);                      \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    if constexpr (::verifyAttr(KIND::NAME))                                    \
      TEST_VERIFY_TRUE(OBJ, NAME);                                             \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), v0);                         \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), v1);                         \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), v2);                         \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), v3);                         \
    EXPECT_EQ(get##ENAME4##From##NAME##Attr(OBJ), v4);                         \
    EXPECT_EQ(get##ENAME5##From##NAME##Attr(OBJ), v5);                         \
    EXPECT_EQ(get##ENAME6##From##NAME##Attr(OBJ), v6);                         \
    EXPECT_EQ(get##ENAME7##From##NAME##Attr(OBJ), v7);                         \
  }                                                                            \
                                                                               \
  remove##NAME##Attr(OBJ);                                                     \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  ::addAttr(OBJ, IRNAME, 8);                                                   \
  TEST_VERIFY_FALSE(OBJ, NAME);                                                \
  TEST_VERIFY_ERR(OBJ, NAME, "missing value of type");                         \
  remove##NAME##Attr(OBJ);

static constexpr llvm::StringRef ll2 = R"(
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

#define DECLS_LOOP(OBJ, LOOP_F, LOOP_G, LIS)                                   \
  DECLS(OBJ);                                                                  \
  std::unique_ptr<Module> m2 = parseIR(getContext(OBJ), ll2);                  \
  Function *f2 = m2->getFunction("f");                                         \
  DominatorTree dtf(*f2);                                                      \
  LoopInfo lif(dtf);                                                           \
  Function *g2 = m2->getFunction("g");                                         \
  DominatorTree dtg(*g2);                                                      \
  LoopInfo lig(dtg);                                                           \
  SmallVector<const LoopInfo *, 4> LIS = {&lig, &lif};                         \
  [[maybe_unused]] Loop *LOOP_F = *lif.begin();                                \
  [[maybe_unused]] Loop *LOOP_G = *lig.begin();

#define TEST_ATTR_LOOP(OBJ, LOOP_F, LOOP_G, LIS, KIND, NAME, IRNAME,           \
                       CUSTOMVERIFY)                                           \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  add##NAME##Attr(OBJ, *LOOP_F);                                               \
  EXPECT_TRUE(has##NAME##Attr(OBJ));                                           \
  if constexpr (::verifyAttr(KIND::NAME))                                      \
    TEST_VERIFY_TRUE(OBJ, NAME);                                               \
  EXPECT_EQ(get##NAME##Attr(OBJ, LIS), LOOP_F);                                \
                                                                               \
  add##NAME##Attr(OBJ, *LOOP_G);                                               \
  EXPECT_TRUE(has##NAME##Attr(OBJ));                                           \
  if constexpr (::verifyAttr(KIND::NAME))                                      \
    TEST_VERIFY_TRUE(OBJ, NAME);                                               \
  EXPECT_EQ(get##NAME##Attr(OBJ, LIS), LOOP_G);                                \
                                                                               \
  remove##NAME##Attr(OBJ);                                                     \
  EXPECT_FALSE(has##NAME##Attr(OBJ));                                          \
  TEST_VERIFY_TRUE(OBJ, NAME);                                                 \
                                                                               \
  ::addAttr(OBJ, IRNAME, 1);                                                   \
  TEST_VERIFY_FALSE(OBJ, NAME);                                                \
  TEST_VERIFY_ERR(OBJ, NAME, "MDNode is not a valid loop id");                 \
  remove##NAME##Attr(OBJ);

#define TEST_ATTR_ATTRS(OBJ)                                                   \
  SmallVector<StringRef> inp = {"attr-1", "attr-3", "attr-7"};                 \
  for (StringRef name : inp)                                                   \
    detail::addAttr(OBJ, name, {});                                            \
                                                                               \
  SmallVector<StringRef> got;                                                  \
  for (const MDNode &attr : detail::attrs(OBJ))                                \
    got.push_back(detail::getRawAttrName(attr));                               \
                                                                               \
  EXPECT_EQ(inp, got);

#endif // KITSUNE_UNITTEST_TEST_ATTRS_COMMON_H
