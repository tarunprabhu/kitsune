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

#include "llvm/ADT/StringRef.h"

namespace llvm {

class GlobalVariable;
class LLVMContext;

/// \addtogroup kitsune
/// @{

/// Get the LLVM context from a global variable. This is useful when generating
/// code from tablegen macros.
LLVMContext &getContext(const GlobalVariable &g);

/// Get the name of a global variable. If the global is unnamed, a string of the
/// form `@<N>` will be returned. This is how the global would appear in
/// human-readable LLVM-IR.
std::string getName(const GlobalVariable &g);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_GV_UTILS_H
