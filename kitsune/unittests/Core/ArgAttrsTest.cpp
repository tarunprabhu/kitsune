//===- ArgAttrsTest.cpp - Unit tests for Kitsune's argument attributes ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ArgAttrs.h"
#include "Core/ArgAttrsImpl.h"
#include "Core/AttrsImpl.h"
#include "Core/VerifierImpl.h"
#include "TestAttrsCommon.h"
#include "kitsune/Core/ArgUtils.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Override get<>() and verifyAttr<> if needed. See documentation in the base
// class for details.
class KitArgAttrs : public TestAttrsBase<KitArgAttrs, ArgAttrKind> {};

TEST_F(KitArgAttrs, attrName) {
#define ARG_ATTR(NAME, IRNAME, ...)                                            \
  EXPECT_EQ(getAttrName(ArgAttrKind::NAME), IRNAME);                           \
  EXPECT_TRUE(getAttrName(ArgAttrKind::NAME).starts_with("kit.arg."));
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"
}

TEST_F(KitArgAttrs, attrKind) {
  EXPECT_EQ(getArgAttrKind("emmanuel"), std::nullopt);
#define ARG_ATTR(NAME, IRNAME, ...)                                            \
  EXPECT_EQ(getArgAttrKind(IRNAME), ArgAttrKind::NAME);
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"
}

TEST_F(KitArgAttrs, verifyGeneric) {
  DECLS;
#define ARG_ATTR_0(NAME, IRNAME, ...)                                          \
  TEST_GENERIC_VERIFY_0(*a, ArgAttrKind, NAME, IRNAME)
#define ARG_ATTR(NAME, IRNAME, ...)                                            \
  TEST_GENERIC_VERIFY_N(*a, ArgAttrKind, NAME, IRNAME)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"
}

TEST_F(KitArgAttrs, attrsGeneric) {
  DECLS;

#define ARG_ATTR_0(NAME, ...) TEST_GENERIC_ATTR_0(*a, ArgAttrKind, NAME)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"

#define ARG_ATTR_0(...)
#define ARG_ATTR(NAME, ...) TEST_GENERIC_ATTR_N(*a, ArgAttrKind, NAME)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"
}

TEST_F(KitArgAttrs, attr0) {
  DECLS;
#define ARG_ATTR_0(...) TEST_ATTR_0(*a, ArgAttrKind, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"
}

TEST_F(KitArgAttrs, attr1) {
  DECLS;
#define ARG_ATTR_1(...) TEST_ATTR_1(*a, ArgAttrKind, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"
}

TEST_F(KitArgAttrs, attr2) {
  DECLS;
#define ARG_ATTR_2(...) TEST_ATTR_2(*a, ArgAttrKind, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"
}

TEST_F(KitArgAttrs, attr3) {
  DECLS;
#define ARG_ATTR_3(...) TEST_ATTR_3(*a, ArgAttrKind, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"
}

TEST_F(KitArgAttrs, attr4) {
  DECLS;
#define ARG_ATTR_4(...) TEST_ATTR_4(*a, ArgAttrKind, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"
}

TEST_F(KitArgAttrs, attr5) {
  DECLS;
#define ARG_ATTR_5(...) TEST_ATTR_5(*a, ArgAttrKind, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"
}

TEST_F(KitArgAttrs, attr6) {
  DECLS;
#define ARG_ATTR_6(...) TEST_ATTR_6(*a, ArgAttrKind, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"
}

TEST_F(KitArgAttrs, attr7) {
  DECLS;
#define ARG_ATTR_7(...) TEST_ATTR_7(*a, ArgAttrKind, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"
}

TEST_F(KitArgAttrs, attr8) {
  DECLS;
#define ARG_ATTR_8(...) TEST_ATTR_8(*a, ArgAttrKind, __VA_ARGS__)
#define GET_ARG_ATTRS
#include "kitsune/Core/ArgAttrs.inc"
}

TEST_F(KitArgAttrs, attrRange) {
  DECLS;
  TEST_ATTR_ATTRS(*a)
}

} // namespace
