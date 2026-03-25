//===- GVUtils.h - Utilities for LLVM GlobalVariables ----------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM GlobalVariable's.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_GV_UTILS_H
#define KITSUNE_CORE_GV_UTILS_H

namespace llvm {

/// \addtogroup kitsune
/// @{

class GlobalVariable;
class LLVMContext;

/// Get the LLVM context from a global variable. This is useful when generating
/// code from tablegen macros.
LLVMContext &getContext(GlobalVariable &g);
LLVMContext &getContext(const GlobalVariable &g);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_GV_UTILS_H
