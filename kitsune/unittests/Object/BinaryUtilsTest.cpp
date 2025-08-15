//===- BinaryUtilsTest.cpp - Tests for Kitsune's binary object utilities --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Object/BinaryUtils.h"
#include "CompressedBinary.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

TEST(BinaryUtilsTest, isArchive) {
  auto check = [](const Binary &bin, bool expected) -> void {
    EXPECT_EQ(isArchive(bin), expected);
    EXPECT_EQ(isArchive(bin.getData()), expected);
    EXPECT_EQ(isArchive(bin.getMemoryBufferRef()), expected);
  };

  check(*arEmpty, true);
  check(*elfExecDynamic, false);
  check(*elfExecStatic, false);
  check(*elfEmpty, false);
  check(*elfShared, false);
  check(*machOExec, false);
  check(*machOObject, false);
  check(*machOShared, false);
}

TEST(BinaryUtilsTest, isExecutable) {
  auto check = [](const Binary &bin, bool expected) -> void {
    EXPECT_EQ(isExecutable(bin), expected);
    EXPECT_EQ(isExecutable(bin.getData()), expected);
    EXPECT_EQ(isExecutable(bin.getMemoryBufferRef()), expected);
  };

  check(*arEmpty, false);
  check(*elfExecDynamic, true);
  check(*elfExecStatic, true);
  check(*elfEmpty, false);
  check(*elfShared, false);
  check(*machOExec, true);
  check(*machOObject, false);
  check(*machOShared, false);
}

TEST(BinaryUtilsTest, isObject) {
  auto check = [](const Binary &bin, bool expected) -> void {
    EXPECT_EQ(isObject(bin), expected);
    EXPECT_EQ(isObject(bin.getData()), expected);
    EXPECT_EQ(isObject(bin.getMemoryBufferRef()), expected);
  };

  check(*arEmpty, false);
  check(*elfExecDynamic, false);
  check(*elfExecStatic, false);
  check(*elfEmpty, true);
  check(*elfShared, false);
  check(*machOExec, false);
  check(*machOObject, true);
  check(*machOShared, false);
}

TEST(BinaryUtilsTest, isShared) {
  auto check = [](const Binary &bin, bool expected) -> void {
    EXPECT_EQ(isShared(bin), expected);
    EXPECT_EQ(isShared(bin.getData()), expected);
    EXPECT_EQ(isShared(bin.getMemoryBufferRef()), expected);
  };

  check(*arEmpty, false);
  check(*elfExecDynamic, false);
  check(*elfExecStatic, false);
  check(*elfEmpty, false);
  check(*elfShared, true);
  check(*machOExec, false);
  check(*machOObject, false);
  check(*machOShared, true);
}
