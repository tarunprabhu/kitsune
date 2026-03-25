//===- TestAttrsCommon.h - Utilities shared by tests for attrs -*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_UNITTEST_TEST_ATTRS_COMMON_H
#define KITSUNE_UNITTEST_TEST_ATTRS_COMMON_H

#include "TestValues.h"

#define TEST_GENERIC_VERIFY_N(OS, OBJ, ENUMCLASS, NAME, IRNAME)                \
  do {                                                                         \
    addMetadata(OBJ, IRNAME, {});                                              \
                                                                               \
    OS.str().clear();                                                          \
    EXPECT_FALSE(verifyAttr(OBJ, ENUMCLASS::NAME));                            \
    EXPECT_FALSE(verifyAttr(OBJ, ENUMCLASS::NAME, &OS));                       \
    EXPECT_TRUE(StringRef(OS.str()).starts_with("Missing value of type"));     \
                                                                               \
    remove##NAME##Attr((OBJ));                                                 \
  } while (0)

#define TEST_GENERIC_VERIFY_0(OS, OBJ, ENUMCLASS, NAME, IRNAME)                \
  do {                                                                         \
    addMetadata(OBJ, IRNAME, MDString::get(ctx, ""));                          \
                                                                               \
    OS.str().clear();                                                          \
    EXPECT_FALSE(verifyAttr(OBJ, ENUMCLASS::NAME));                            \
    EXPECT_FALSE(verifyAttr(OBJ, ENUMCLASS::NAME, &OS));                       \
    EXPECT_TRUE(                                                               \
        StringRef(OS.str()).starts_with("Unexpected value in attribute"));     \
                                                                               \
    remove##NAME##Attr(OBJ);                                                   \
  } while (0)

#define TEST_GENERIC_ATTR_0(OBJ, ENUMCLASS, NAME)                              \
  do {                                                                         \
    EXPECT_FALSE(hasAttr(OBJ, ENUMCLASS::NAME));                               \
    addAttr(OBJ, ENUMCLASS::NAME);                                             \
    EXPECT_TRUE(hasAttr(OBJ, ENUMCLASS::NAME));                                \
    addAttr(OBJ, ENUMCLASS::NAME);                                             \
    EXPECT_TRUE(hasAttr(OBJ, ENUMCLASS::NAME));                                \
    removeAttr(OBJ, ENUMCLASS::NAME);                                          \
    EXPECT_FALSE(hasAttr(OBJ, ENUMCLASS::NAME));                               \
  } while (0)

#define TEST_GENERIC_ATTR_N(OBJ, ENUMCLASS, NAME)                              \
  do {                                                                         \
    EXPECT_EXIT(addAttr(OBJ, ENUMCLASS::NAME), ::testing::ExitedWithCode(1),   \
                "error: cannot add attribute");                                \
  } while (0)

#define TEST_ATTR_0(OS, OBJ, NAME, IRNAME)                                     \
  do {                                                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_FALSE(has##NAME##Attr(OBJ));                                        \
                                                                               \
    add##NAME##Attr(OBJ);                                                      \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
                                                                               \
    add##NAME##Attr(OBJ);                                                      \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
                                                                               \
    remove##NAME##Attr(OBJ);                                                   \
    EXPECT_FALSE(has##NAME##Attr(OBJ));                                        \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
                                                                               \
    OS.str().clear();                                                          \
    addMetadata(OBJ, IRNAME, 0);                                               \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_TRUE(verify##NAME##Attr(OBJ, &os));                                 \
    EXPECT_TRUE(OS.str().empty());                                             \
    remove##NAME##Attr(OBJ);                                                   \
  } while (0)

#define TEST_ATTR_1(OS, OBJ, NAME, IRNAME, TYPE)                               \
  do {                                                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_FALSE(has##NAME##Attr(OBJ));                                        \
                                                                               \
    add##NAME##Attr(OBJ, get<TYPE>(0));                                        \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_EQ(get##NAME##Attr(OBJ), get<TYPE>(0));                             \
                                                                               \
    add##NAME##Attr(OBJ, get<TYPE>(1));                                        \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_EQ(get##NAME##Attr(OBJ), get<TYPE>(1));                             \
                                                                               \
    remove##NAME##Attr(OBJ);                                                   \
    EXPECT_FALSE(has##NAME##Attr(OBJ));                                        \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
                                                                               \
    OS.str().clear();                                                          \
    addMetadata(OBJ, IRNAME, 1);                                               \
    EXPECT_FALSE(verify##NAME##Attr(OBJ));                                     \
    EXPECT_FALSE(verify##NAME##Attr(OBJ, &OS));                                \
    EXPECT_TRUE(StringRef(OS.str()).starts_with("Missing value of type"));     \
    remove##NAME##Attr(OBJ);                                                   \
  } while (0)

#define TEST_ATTR_2(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,    \
                    EN1)                                                       \
  do {                                                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_FALSE(has##NAME##Attr(OBJ));                                        \
                                                                               \
    add##NAME##Attr(OBJ, get<ETY0>(0), get<ETY1>(1));                          \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), get<ETY0>(0));               \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), get<ETY1>(1));               \
                                                                               \
    add##NAME##Attr(OBJ, get<ETY0>(1), get<ETY1>(0));                          \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), get<ETY0>(1));               \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), get<ETY1>(0));               \
                                                                               \
    remove##NAME##Attr(OBJ);                                                   \
    EXPECT_FALSE(has##NAME##Attr(OBJ));                                        \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
                                                                               \
    OS.str().clear();                                                          \
    addMetadata(OBJ, IRNAME, 2);                                               \
    EXPECT_FALSE(verify##NAME##Attr(OBJ));                                     \
    EXPECT_FALSE(verify##NAME##Attr(OBJ, &os));                                \
    EXPECT_TRUE(StringRef(OS.str()).starts_with("Missing value of type"));     \
    remove##NAME##Attr(OBJ);                                                   \
  } while (0)

#define TEST_ATTR_3(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,    \
                    EN1, ETY2, ENAME2, EN2)                                    \
  do {                                                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_FALSE(has##NAME##Attr(OBJ));                                        \
                                                                               \
    add##NAME##Attr(OBJ, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2));            \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), get<ETY0>(0));               \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), get<ETY1>(1));               \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), get<ETY2>(2));               \
                                                                               \
    add##NAME##Attr(OBJ, get<ETY0>(2), get<ETY1>(1), get<ETY2>(0));            \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), get<ETY0>(2));               \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), get<ETY1>(1));               \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), get<ETY2>(0));               \
                                                                               \
    remove##NAME##Attr(OBJ);                                                   \
    EXPECT_FALSE(has##NAME##Attr(OBJ));                                        \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
                                                                               \
    OS.str().clear();                                                          \
    addMetadata(OBJ, IRNAME, 3);                                               \
    EXPECT_FALSE(verify##NAME##Attr(OBJ));                                     \
    EXPECT_FALSE(verify##NAME##Attr(OBJ, &OS));                                \
    EXPECT_TRUE(StringRef(OS.str()).starts_with("Missing value of type"));     \
    remove##NAME##Attr(OBJ);                                                   \
  } while (0)

#define TEST_ATTR_4(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,    \
                    EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3)                 \
  do {                                                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_FALSE(has##NAME##Attr(OBJ));                                        \
                                                                               \
    add##NAME##Attr(OBJ, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2),             \
                    get<ETY3>(3));                                             \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), get<ETY0>(0));               \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), get<ETY1>(1));               \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), get<ETY2>(2));               \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), get<ETY3>(3));               \
                                                                               \
    add##NAME##Attr(OBJ, get<ETY0>(3), get<ETY1>(2), get<ETY2>(1),             \
                    get<ETY3>(0));                                             \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), get<ETY0>(3));               \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), get<ETY1>(2));               \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), get<ETY2>(1));               \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), get<ETY3>(0));               \
                                                                               \
    remove##NAME##Attr(OBJ);                                                   \
    EXPECT_FALSE(has##NAME##Attr(OBJ));                                        \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
                                                                               \
    OS.str().clear();                                                          \
    addMetadata(OBJ, IRNAME, 4);                                               \
    EXPECT_FALSE(verify##NAME##Attr(OBJ));                                     \
    EXPECT_FALSE(verify##NAME##Attr(OBJ, &OS));                                \
    EXPECT_TRUE(StringRef(OS.str()).starts_with("Missing value of type"));     \
    remove##NAME##Attr(OBJ);                                                   \
  } while (0)

#define TEST_ATTR_5(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,    \
                    EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4,   \
                    EN4)                                                       \
  do {                                                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_FALSE(has##NAME##Attr(OBJ));                                        \
                                                                               \
    add##NAME##Attr(OBJ, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2),             \
                    get<ETY3>(3), get<ETY4>(4));                               \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), get<ETY0>(0));               \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), get<ETY1>(1));               \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), get<ETY2>(2));               \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), get<ETY3>(3));               \
    EXPECT_EQ(get##ENAME4##From##NAME##Attr(OBJ), get<ETY4>(4));               \
                                                                               \
    add##NAME##Attr(OBJ, get<ETY0>(4), get<ETY1>(3), get<ETY2>(2),             \
                    get<ETY3>(1), get<ETY4>(0));                               \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), get<ETY0>(4));               \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), get<ETY1>(3));               \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), get<ETY2>(2));               \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), get<ETY3>(1));               \
    EXPECT_EQ(get##ENAME4##From##NAME##Attr(OBJ), get<ETY4>(0));               \
                                                                               \
    remove##NAME##Attr(OBJ);                                                   \
    EXPECT_FALSE(has##NAME##Attr(OBJ));                                        \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
                                                                               \
    OS.str().clear();                                                          \
    addMetadata(OBJ, IRNAME, 5);                                               \
    EXPECT_FALSE(verify##NAME##Attr(OBJ));                                     \
    EXPECT_FALSE(verify##NAME##Attr(OBJ, &OS));                                \
    EXPECT_TRUE(StringRef(OS.str()).starts_with("Missing value of type"));     \
    remove##NAME##Attr(OBJ);                                                   \
  } while (0)

#define TEST_ATTR_6(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,    \
                    EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4,   \
                    EN4, ETY5, ENAME5, EN5)                                    \
  do {                                                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_FALSE(has##NAME##Attr(OBJ));                                        \
                                                                               \
    add##NAME##Attr(OBJ, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2),             \
                    get<ETY3>(3), get<ETY4>(4), get<ETY5>(5));                 \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), get<ETY0>(0));               \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), get<ETY1>(1));               \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), get<ETY2>(2));               \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), get<ETY3>(3));               \
    EXPECT_EQ(get##ENAME4##From##NAME##Attr(OBJ), get<ETY4>(4));               \
    EXPECT_EQ(get##ENAME5##From##NAME##Attr(OBJ), get<ETY5>(5));               \
                                                                               \
    add##NAME##Attr(OBJ, get<ETY0>(5), get<ETY1>(4), get<ETY2>(3),             \
                    get<ETY3>(2), get<ETY4>(1), get<ETY5>(0));                 \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), get<ETY0>(5));               \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), get<ETY1>(4));               \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), get<ETY2>(3));               \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), get<ETY3>(2));               \
    EXPECT_EQ(get##ENAME4##From##NAME##Attr(OBJ), get<ETY4>(1));               \
    EXPECT_EQ(get##ENAME5##From##NAME##Attr(OBJ), get<ETY5>(0));               \
                                                                               \
    remove##NAME##Attr(OBJ);                                                   \
    EXPECT_FALSE(has##NAME##Attr(OBJ));                                        \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
                                                                               \
    OS.str().clear();                                                          \
    addMetadata(OBJ, IRNAME, 6);                                               \
    EXPECT_FALSE(verify##NAME##Attr(OBJ));                                     \
    EXPECT_FALSE(verify##NAME##Attr(OBJ, &OS));                                \
    EXPECT_TRUE(StringRef(OS.str()).starts_with("Missing value of type"));     \
    remove##NAME##Attr(OBJ);                                                   \
  } while (0)

#define TEST_ATTR_7(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,    \
                    EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4,   \
                    EN4, ETY5, ENAME5, EN5, ETY6, ENAME6, EN6)                 \
  do {                                                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_FALSE(has##NAME##Attr(OBJ));                                        \
                                                                               \
    add##NAME##Attr(OBJ, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2),             \
                    get<ETY3>(3), get<ETY4>(4), get<ETY5>(5), get<ETY6>(6));   \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), get<ETY0>(0));               \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), get<ETY1>(1));               \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), get<ETY2>(2));               \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), get<ETY3>(3));               \
    EXPECT_EQ(get##ENAME4##From##NAME##Attr(OBJ), get<ETY4>(4));               \
    EXPECT_EQ(get##ENAME5##From##NAME##Attr(OBJ), get<ETY5>(5));               \
    EXPECT_EQ(get##ENAME6##From##NAME##Attr(OBJ), get<ETY6>(6));               \
                                                                               \
    add##NAME##Attr(OBJ, get<ETY0>(6), get<ETY1>(5), get<ETY2>(4),             \
                    get<ETY3>(3), get<ETY4>(2), get<ETY5>(1), get<ETY6>(0));   \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), get<ETY0>(6));               \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), get<ETY1>(5));               \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), get<ETY2>(4));               \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), get<ETY3>(3));               \
    EXPECT_EQ(get##ENAME4##From##NAME##Attr(OBJ), get<ETY4>(2));               \
    EXPECT_EQ(get##ENAME5##From##NAME##Attr(OBJ), get<ETY5>(1));               \
    EXPECT_EQ(get##ENAME6##From##NAME##Attr(OBJ), get<ETY6>(0));               \
                                                                               \
    remove##NAME##Attr(OBJ);                                                   \
    EXPECT_FALSE(has##NAME##Attr(OBJ));                                        \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
                                                                               \
    OS.str().clear();                                                          \
    addMetadata(OBJ, IRNAME, 7);                                               \
    EXPECT_FALSE(verify##NAME##Attr(OBJ));                                     \
    EXPECT_FALSE(verify##NAME##Attr(OBJ, &OS));                                \
    EXPECT_TRUE(StringRef(OS.str()).starts_with("Missing value of type"));     \
    remove##NAME##Attr(OBJ);                                                   \
  } while (0)

#define TEST_ATTR_8(OS, OBJ, NAME, IRNAME, ETY0, ENAME0, EN0, ETY1, ENAME1,    \
                    EN1, ETY2, ENAME2, EN2, ETY3, ENAME3, EN3, ETY4, ENAME4,   \
                    EN4, ETY5, ENAME5, EN5, ETY6, ENAME6, EN6, ETY7, ENAME7,   \
                    EN7)                                                       \
  do {                                                                         \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_FALSE(has##NAME##Attr(OBJ));                                        \
                                                                               \
    add##NAME##Attr(OBJ, get<ETY0>(0), get<ETY1>(1), get<ETY2>(2),             \
                    get<ETY3>(3), get<ETY4>(4), get<ETY5>(5), get<ETY6>(6),    \
                    get<ETY7>(7));                                             \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), get<ETY0>(0));               \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), get<ETY1>(1));               \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), get<ETY2>(2));               \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), get<ETY3>(3));               \
    EXPECT_EQ(get##ENAME4##From##NAME##Attr(OBJ), get<ETY4>(4));               \
    EXPECT_EQ(get##ENAME5##From##NAME##Attr(OBJ), get<ETY5>(5));               \
    EXPECT_EQ(get##ENAME6##From##NAME##Attr(OBJ), get<ETY6>(6));               \
    EXPECT_EQ(get##ENAME7##From##NAME##Attr(OBJ), get<ETY7>(7));               \
                                                                               \
    add##NAME##Attr(OBJ, get<ETY0>(7), get<ETY1>(6), get<ETY2>(5),             \
                    get<ETY3>(4), get<ETY4>(3), get<ETY5>(2), get<ETY6>(1),    \
                    get<ETY7>(0));                                             \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
    EXPECT_TRUE(has##NAME##Attr(OBJ));                                         \
    EXPECT_EQ(get##ENAME0##From##NAME##Attr(OBJ), get<ETY0>(7));               \
    EXPECT_EQ(get##ENAME1##From##NAME##Attr(OBJ), get<ETY1>(6));               \
    EXPECT_EQ(get##ENAME2##From##NAME##Attr(OBJ), get<ETY2>(5));               \
    EXPECT_EQ(get##ENAME3##From##NAME##Attr(OBJ), get<ETY3>(4));               \
    EXPECT_EQ(get##ENAME4##From##NAME##Attr(OBJ), get<ETY4>(3));               \
    EXPECT_EQ(get##ENAME5##From##NAME##Attr(OBJ), get<ETY5>(2));               \
    EXPECT_EQ(get##ENAME6##From##NAME##Attr(OBJ), get<ETY6>(1));               \
    EXPECT_EQ(get##ENAME7##From##NAME##Attr(OBJ), get<ETY7>(0));               \
                                                                               \
    remove##NAME##Attr(OBJ);                                                   \
    EXPECT_FALSE(has##NAME##Attr(OBJ));                                        \
    EXPECT_TRUE(verify##NAME##Attr(OBJ));                                      \
                                                                               \
    OS.str().clear();                                                          \
    addMetadata(OBJ, IRNAME, 8);                                               \
    EXPECT_FALSE(verify##NAME##Attr(OBJ));                                     \
    EXPECT_FALSE(verify##NAME##Attr(OBJ, &OS));                                \
    EXPECT_TRUE(StringRef(OS.str()).starts_with("Missing value of type"));     \
    remove##NAME##Attr(OBJ);                                                   \
  } while (0)

#endif // KITSUNE_UNITTEST_TEST_ATTRS_COMMON_H
