//===- TestUtils.h - Utilities shared by core unittests --------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_UNITTESTS_CORE_TEST_UTILS_H
#define KITSUNE_UNITTESTS_CORE_TEST_UTILS_H

#include "llvm/IR/Module.h"

// Parse the IR in the string \p ir.
std::unique_ptr<llvm::Module> parseIR(llvm::LLVMContext &ctx,
                                      llvm::StringRef ir);

#endif // KITSUNE_UNITTESTS_CORE_TEST_UTILS_H
