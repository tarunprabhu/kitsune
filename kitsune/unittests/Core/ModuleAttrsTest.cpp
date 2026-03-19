//===- ModuleAttrsTest.cpp - Unit tests for Kitsune's module attributes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ModuleAttrs.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitModuleAttrs, attrName) {
#define MODULE_ATTR(NAME, IRNAME)                                              \
  EXPECT_EQ(getAttrName(ModuleAttrKind::NAME), IRNAME);                        \
  EXPECT_TRUE(getAttrName(ModuleAttrKind::NAME).starts_with("kit.module."));
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

TEST(KitModuleAttrs, attrKind) {
#define MODULE_ATTR(NAME, IRNAME)                                              \
  EXPECT_EQ(getModuleAttrKind(IRNAME), ModuleAttrKind::NAME);
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
  EXPECT_EQ(getModuleAttrKind("whoops"), std::nullopt);
}

TEST(KitModuleAttrs, attr0) {
  LLVMContext ctx;
  Module m("", ctx);

#define MODULE_ATTR_0(NAME, IRNAME)                                            \
  EXPECT_FALSE(hasAttr(m, ModuleAttrKind::NAME));                              \
  EXPECT_FALSE(has##NAME##Attr(m));                                            \
                                                                               \
  add##NAME##Attr(m);                                                          \
  EXPECT_TRUE(hasAttr(m, ModuleAttrKind::NAME));                               \
  EXPECT_TRUE(has##NAME##Attr(m));                                             \
                                                                               \
  remove##NAME##Attr(m);                                                       \
  EXPECT_FALSE(hasAttr(m, ModuleAttrKind::NAME));                              \
  EXPECT_FALSE(has##NAME##Attr(m));                                            \
                                                                               \
  add##NAME##Attr(m);                                                          \
  removeAttr(m, ModuleAttrKind::NAME);                                         \
  EXPECT_FALSE(hasAttr(m, ModuleAttrKind::NAME));                              \
  EXPECT_FALSE(has##NAME##Attr(m));
#define GET_MODULE_ATTRS
#include "kitsune/Core/ModuleAttrs.inc"
}

// TODO: When adding a module attribute with 1 argument, add a test for it here.
// TEST(KitModuleAttrs, attr1) {
// }

TEST(KitModuleAttrs, attr2) {
  LLVMContext ctx;
  Module m("", ctx);

  EXPECT_FALSE(hasDeviceModuleFlagsAttr(m));
  EXPECT_EQ(getTTIDFromDeviceModuleFlagsAttr(m), std::nullopt);
  EXPECT_EQ(getNameFromDeviceModuleFlagsAttr(m), std::nullopt);

  addDeviceModuleFlagsAttr(m, TTID::Cuda, "brasenose");
  EXPECT_TRUE(hasDeviceModuleFlagsAttr(m));
  EXPECT_EQ(getTTIDFromDeviceModuleFlagsAttr(m), TTID::Cuda);
  EXPECT_EQ(getNameFromDeviceModuleFlagsAttr(m), "brasenose");

  // Adding the same attribute when it already exists should override the
  // existing attribute.
  addDeviceModuleFlagsAttr(m, TTID::Hip, "pembroke");
  EXPECT_TRUE(hasDeviceModuleFlagsAttr(m));
  EXPECT_EQ(getTTIDFromDeviceModuleFlagsAttr(m), TTID::Hip);
  EXPECT_EQ(getNameFromDeviceModuleFlagsAttr(m), "pembroke");

  removeDeviceModuleFlagsAttr(m);
  EXPECT_FALSE(hasDeviceModuleFlagsAttr(m));
  EXPECT_EQ(getTTIDFromDeviceModuleFlagsAttr(m), std::nullopt);
  EXPECT_EQ(getNameFromDeviceModuleFlagsAttr(m), std::nullopt);

  // Removing an attribute that does not exist should have no effect.
  removeDeviceModuleFlagsAttr(m);
  EXPECT_FALSE(hasDeviceModuleFlagsAttr(m));
  EXPECT_EQ(getTTIDFromDeviceModuleFlagsAttr(m), std::nullopt);
  EXPECT_EQ(getNameFromDeviceModuleFlagsAttr(m), std::nullopt);
}

// TODO: When adding a module attribute with 3 arguments, add a test for it.
// TEST(KitModuleAttrs, attr3) {
// }

// TODO: When adding a module attribute with 4 arguments, add a test for it.
// TEST(KitModuleAttrs, attr4) {
// }

// TODO: When adding a module attribute with 5 arguments, add a test for it.
// TEST(KitModuleAttrs, attr5) {
// }

// TODO: When adding a module attribute with 6 arguments, add a test for it.
// TEST(KitModuleAttrs, attr6) {
// }

// TODO: When adding a module attribute with 7 arguments, add a test for it.
// TEST(KitModuleAttrs, attr7) {
// }

// TODO: When adding a module attribute with 8 arguments, add a test for it.
// TEST(KitModuleAttrs, attr8) {
// }

} // namespace
