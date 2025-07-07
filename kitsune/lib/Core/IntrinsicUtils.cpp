//===- IntrinsicUtils.cpp - Utilities for Kitsune's intrinsics ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for Kitsune's intrinsics
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/IntrinsicUtils.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;

bool llvm::isKitsuneIntrinsic(Intrinsic::ID id) {
  return Intrinsic::getBaseName(id).starts_with("llvm.kit.");
}

bool llvm::isKitsuneIntrinsicAsync(Intrinsic::ID id) {
  assert(isKitsuneIntrinsic(id) && "Must be a kitsune intrinsic");
  return Intrinsic::getBaseName(id).starts_with("llvm.kit.async.");
}

bool llvm::isKitsuneIntrinsicBlocking(Intrinsic::ID id) {
  return not isKitsuneIntrinsicAsync(id);
}
