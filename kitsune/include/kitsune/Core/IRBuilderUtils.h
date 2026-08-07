//===- IRBuilderUtils.h - Utilities for the IRBuilder ----------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM's IRBuilder.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_IR_BUILDER_UTILS_H
#define KITSUNE_CORE_IR_BUILDER_UTILS_H

#include "kitsune/Core/LibFuncs.h"
#include "llvm/IR/IRBuilder.h"

namespace llvm {

class Function;
class Module;

/// \addtogroup kitsune
/// @{

/// If the insert point of the builder is set to a basic block in a function,
/// return the function. In all other cases, return nullptr.
Function *getFunction(IRBuilder<> &builder);

/// If the insert point of the builder is set to a basic block in a function,
/// and that function is in a module, return the module. In all other cases,
/// return nullptr.
Module *getModule(IRBuilder<> &builder);

/// Insert a call to the library function \p func with arguments \p args using
/// the builder \p builder. The insert point of \p builder must be set to a
/// basic block contained in a module.
Value *createCall(IRBuilder<> &builder, KitFunc f, ArrayRef<Value *> args = {},
                  StringRef name = "");

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_IR_BUILDER_UTILS_H
