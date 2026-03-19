//===- DiagnosticsTest.cpp - Unit tests for Kitsune's diagnostics ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/Diagnostics.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitDiagnostics, isError) {
#define GET_DIAGS
#define DIAG_ERROR(NAME, MSG) EXPECT_TRUE(isError(DiagID::NAME));
#define DIAG_WARNING(NAME, MSG) EXPECT_FALSE(isError(DiagID::NAME));
#define DIAG_REMARK(NAME, MSG) EXPECT_FALSE(isError(DiagID::NAME));
#define DIAG_NOTE(NAME, MSG) EXPECT_FALSE(isError(DiagID::NAME));
#include "kitsune/Support/Diagnostics.inc"
}

TEST(KitDiagnostics, isWarning) {
#define GET_DIAGS
#define DIAG_ERROR(NAME, MSG) EXPECT_FALSE(isWarning(DiagID::NAME));
#define DIAG_WARNING(NAME, MSG) EXPECT_TRUE(isWarning(DiagID::NAME));
#define DIAG_REMARK(NAME, MSG) EXPECT_FALSE(isWarning(DiagID::NAME));
#define DIAG_NOTE(NAME, MSG) EXPECT_FALSE(isWarning(DiagID::NAME));
#include "kitsune/Support/Diagnostics.inc"
}

TEST(KitDiagnostics, isRemark) {
#define GET_DIAGS
#define DIAG_ERROR(NAME, MSG) EXPECT_FALSE(isRemark(DiagID::NAME));
#define DIAG_WARNING(NAME, MSG) EXPECT_FALSE(isRemark(DiagID::NAME));
#define DIAG_REMARK(NAME, MSG) EXPECT_TRUE(isRemark(DiagID::NAME));
#define DIAG_NOTE(NAME, MSG) EXPECT_FALSE(isRemark(DiagID::NAME));
#include "kitsune/Support/Diagnostics.inc"
}

TEST(KitDiagnostics, isNote) {
#define GET_DIAGS
#define DIAG_ERROR(NAME, MSG) EXPECT_FALSE(isNote(DiagID::NAME));
#define DIAG_WARNING(NAME, MSG) EXPECT_FALSE(isNote(DiagID::NAME));
#define DIAG_REMARK(NAME, MSG) EXPECT_FALSE(isNote(DiagID::NAME));
#define DIAG_NOTE(NAME, MSG) EXPECT_TRUE(isNote(DiagID::NAME));
#include "kitsune/Support/Diagnostics.inc"
}

} // namespace
