//===- StringUtils.cpp - String utilities ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// String utilities for kitsune. Some of these are tweaks to the utilities
// provided by LLVM.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/StringUtils.h"

using namespace llvm;

void llvm::detail::sjoin(raw_ostream &os) {}
