//===- ObjectUtilsTest.cpp - Tests for LLVM's binary object utilities -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Object/ObjectUtils.h"
#include "llvm/Object/ObjectFile.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

