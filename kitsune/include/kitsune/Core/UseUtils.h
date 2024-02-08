//===- UseUtils.h - Utilities for LLVM's Use objects -----------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM's Use objects.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_USE_UTILS_H
#define KITSUNE_CORE_USE_UTILS_H

namespace llvm {

class BasicBlock;
class Use;

/// \addtogroup kitsune
/// @{

/// Check if the user of the given use, \p use, is an instruction in the basic
/// block \p bb.
bool isUseInBlock(Use &use, BasicBlock &bb);

/// Check if the user of the given use, \p use, is a constant. This will
/// typically be the initializer of a global variable, but it need not be.
bool isUseInConstant(Use &use);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_USE_UTILS_H
