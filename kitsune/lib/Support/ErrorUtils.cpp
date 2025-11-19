//===- ErrorUtils.cpp - Utilities for LLVM's error objects ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM's error objects
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/ErrorUtils.h"
#include "llvm/Support/Error.h"

void llvm::ignoreAllErrors(Error &&err) {
  handleAllErrors(std::move(err), [](const ErrorInfoBase &) {});
}
