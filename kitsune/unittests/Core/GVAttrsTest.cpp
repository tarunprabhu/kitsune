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
#include "Core/VerifierImpl.h"
#include "TestAttrsCommon.h"
#include "kitsune/Core/GVUtils.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Override get<>() and verifyAttr<> if needed. See documentation in the base
// class for details.
class KitGVAttrs : public TestAttrsBase<KitGVAttrs, GVAttrKind> {
public:
  static constexpr bool verifyAttr(GVAttrKind attr) {
    switch (attr) {
    case GVAttrKind::BitCode:
    case GVAttrKind::DeviceCode:
      return false;
    default:
      return true;
    }
  }
};

TEST_F(KitGVAttrs, attrName) {
#define GV_ATTR(NAME, IRNAME, ...)                                             \
  EXPECT_EQ(getAttrName(GVAttrKind::NAME), IRNAME);                            \
  EXPECT_TRUE(getAttrName(GVAttrKind::NAME).starts_with("kit.gv."));
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST_F(KitGVAttrs, attrKind) {
  EXPECT_EQ(getGVAttrKind("brasenose"), std::nullopt);
#define GV_ATTR(NAME, IRNAME, ...)                                             \
  EXPECT_EQ(getGVAttrKind(IRNAME), GVAttrKind::NAME);
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST_F(KitGVAttrs, verifyGeneric) {
  DECLS;
#define GV_ATTR_0(NAME, IRNAME, ...)                                           \
  TEST_GENERIC_VERIFY_0(*g, GVAttrKind, NAME, IRNAME)
#define GV_ATTR(NAME, IRNAME, ...)                                             \
  TEST_GENERIC_VERIFY_N(*g, GVAttrKind, NAME, IRNAME)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST_F(KitGVAttrs, attrsGeneric) {
  DECLS;

#define GV_ATTR_0(NAME, ...) TEST_GENERIC_ATTR_0(*g, GVAttrKind, NAME)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_0(...)
#define GV_ATTR(NAME, ...) TEST_GENERIC_ATTR_N(*g, GVAttrKind, NAME)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST_F(KitGVAttrs, attr0) {
  DECLS;
#define GV_ATTR_0(...) TEST_ATTR_0(*g, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST_F(KitGVAttrs, attr1) {
  DECLS;
#define GV_ATTR_1(...) TEST_ATTR_1(*g, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST_F(KitGVAttrs, attr2) {
  DECLS;
#define GV_ATTR_2(...) TEST_ATTR_2(*g, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST_F(KitGVAttrs, attr3) {
  DECLS;
#define GV_ATTR_3(...) TEST_ATTR_3(*g, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST_F(KitGVAttrs, attr4) {
  DECLS;
#define GV_ATTR_4(...) TEST_ATTR_4(*g, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST_F(KitGVAttrs, attr5) {
  DECLS;
#define GV_ATTR_5(...) TEST_ATTR_5(*g, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST_F(KitGVAttrs, attr6) {
  DECLS;
#define GV_ATTR_6(...) TEST_ATTR_6(*g, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST_F(KitGVAttrs, attr7) {
  DECLS;
#define GV_ATTR_7(...) TEST_ATTR_7(*g, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST_F(KitGVAttrs, attr8) {
  DECLS;
#define GV_ATTR_8(...) TEST_ATTR_8(*g, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

TEST_F(KitGVAttrs, attrRange) {
  DECLS;
  TEST_ATTR_ATTRS(*g)
}

} // namespace
