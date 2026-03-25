//===- InstUtils.h - Utilities for LLVM instructions -----------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM instructions.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_INST_UTILS_H
#define KITSUNE_CORE_INST_UTILS_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

class Instruction;
class LLVMContext;

/// \addtogroup kitsune
/// @{

/// Get the LLVM context from an instruction. This is useful when generating
/// code from tablegen macros.
LLVMContext &getContext(Instruction &inst);
LLVMContext &getContext(const Instruction &inst);

/// Get the class name of the given instruction. This will be a string literal
/// representing the C++ name of the concrete class of the instruction. For
/// instance, "BinaryOperator", "ICmpInst", "FCmpInst", "DetachInst" etc.
StringRef getInstClassName(const Instruction &inst);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_INST_UTILS_H
