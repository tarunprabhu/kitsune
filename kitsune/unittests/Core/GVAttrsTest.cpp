//=- GVAttrsTest.cpp - Unit tests for Kitsune's global variable attributes --=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/GVAttrs.h"
#include "Core/AttrsImpl.h"
#include "Core/GVAttrsImpl.h"
#include "TestAttrsCommon.h"
#include "kitsune/Core/AttrsCommon.h"
#include "kitsune/Core/GVUtils.h"
#include "kitsune/Core/Verifier.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

template <typename T, GVAttrKind Attr> static T get(unsigned idx) {
  return get_<T>(idx);
}

// In some cases, it is difficult to construct a valid attribute - for instance
// if the attribute initializer must be valid bitcode. In such cases, we test
// everything but the verifier. lit tests must be added to ensure that the
// verification works correctly.
[[maybe_unused]]
static constexpr bool verifyAttr(GVAttrKind attr) {
  switch (attr) {
  case GVAttrKind::BitCode:
  case GVAttrKind::DeviceCode:
    return false;
  default:
    return true;
  }
}

TEST(KitGVAttrs, attrName) {
#define GV_ATTR(NAME, IRNAME, ...)                                             \
  EXPECT_EQ(getAttrName(GVAttrKind::NAME), IRNAME);                            \
  EXPECT_TRUE(getAttrName(GVAttrKind::NAME).starts_with("kit.gv."));
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attrKind) {
  EXPECT_EQ(getGVAttrKind("brasenose"), std::nullopt);
#define GV_ATTR(NAME, IRNAME, ...)                                             \
  EXPECT_EQ(getGVAttrKind(IRNAME), GVAttrKind::NAME);
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

#define DECLS(OBJ)                                                             \
  std::string buf;                                                             \
  raw_string_ostream OS(buf);                                                  \
  [[maybe_unused]] KitVerifier VOS(&OS);                                       \
  [[maybe_unused]] KitVerifier VNULL;                                          \
  LLVMContext ctx;                                                             \
  Module m("", ctx);                                                           \
  Type *i32 = Type::getInt32Ty(ctx);                                           \
  GlobalVariable OBJ = m.getOrInsertGlobal("g", i32);                          \
  (OBJ).setInitializer(Constant::getNullValue(i32));

TEST(KitGVAttrs, verifyGeneric) {
  DECLS(*g);
#define GV_ATTR_0(NAME, IRNAME, ...)                                           \
  TEST_GENERIC_VERIFY_0(*g, GVAttrKind, NAME, IRNAME)
#define GV_ATTR(NAME, IRNAME, ...)                                             \
  TEST_GENERIC_VERIFY_N(*g, GVAttrKind, NAME, IRNAME)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attrsGeneric) {
  DECLS(*g);

#define GV_ATTR_0(NAME, ...) TEST_GENERIC_ATTR_0(*g, GVAttrKind, NAME)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_0(...)
#define GV_ATTR(NAME, ...) TEST_GENERIC_ATTR_N(*g, GVAttrKind, NAME)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr0) {
  DECLS(*g);
#define GV_ATTR_0(...) TEST_ATTR_0(*g, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr1) {
  DECLS(*g);
#define GV_ATTR_1(...) TEST_ATTR_1(*g, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr2) {
  DECLS(*g);
#define GV_ATTR_2(...) TEST_ATTR_2(*g, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr3) {
  DECLS(*g);
#define GV_ATTR_3(...) TEST_ATTR_3(*g, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr4) {
  DECLS(*g);
#define GV_ATTR_4(...) TEST_ATTR_4(*g, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr5) {
  DECLS(*g);
#define GV_ATTR_5(...) TEST_ATTR_5(*g, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr6) {
  DECLS(*g);
#define GV_ATTR_6(...) TEST_ATTR_6(*g, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr7) {
  DECLS(*g);
#define GV_ATTR_7(...) TEST_ATTR_7(*g, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attr8) {
  DECLS(*g);
#define GV_ATTR_8(...) TEST_ATTR_8(*g, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attrLoop) {
  DECLS_LOOP(*g, loopF, loopG, lis);
#define GV_ATTR_LOOP(...)                                                      \
  TEST_ATTR_LOOP(*g, loopF, loopG, lis, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST(KitGVAttrs, attrRange) {
  DECLS(*g);
  TEST_ATTR_ATTRS(*g)
}

} // namespace
