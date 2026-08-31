//===- GPUMemUtilsTest.cpp - Tests for GPU memory utilities ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/GPUMemUtils.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitGPUMemUtils, toInt) {
  EXPECT_EQ((int)GPUDynMemAllocKind::Global, 1);
  EXPECT_EQ((int)GPUDynMemAllocKind::UVM, 2);
}

TEST(KitGPUMemUtils, toString) {
  EXPECT_EQ(toString(GPUDynMemAllocKind::Global), "global");
  EXPECT_EQ(toString(GPUDynMemAllocKind::UVM), "uvm");
}

TEST(KitGPUMemUtils, fromInt) {
  EXPECT_FALSE(fromInt<GPUDynMemAllocKind>(0));
  EXPECT_EQ(fromInt<GPUDynMemAllocKind>(1), GPUDynMemAllocKind::Global);
  EXPECT_EQ(fromInt<GPUDynMemAllocKind>(2), GPUDynMemAllocKind::UVM);
  EXPECT_FALSE(fromInt<GPUDynMemAllocKind>(3));
}

TEST(KitGPUMemUtils, fromString) {
  EXPECT_FALSE(fromString<GPUDynMemAllocKind>(""));
  EXPECT_FALSE(fromString<GPUDynMemAllocKind>("UVM"));

  EXPECT_EQ(fromString<GPUDynMemAllocKind>("global"),
            GPUDynMemAllocKind::Global);
  EXPECT_EQ(fromString<GPUDynMemAllocKind>("uvm"), GPUDynMemAllocKind::UVM);
}

} // namespace
