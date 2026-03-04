//===- InstructionUtils.h - Utilities for LLVM instructions-----*- C++ -*--===//
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

#ifndef KITSUNE_CORE_INSTRUCTION_UTILS_H
#define KITSUNE_CORE_INSTRUCTION_UTILS_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

/// \addtogroup kitsune
/// @{

class Instruction;

/// Get the class name of the given instruction. This will be a string literal
/// representing the C++ name of the concrete class of the instruction. For
/// instance, "BinaryOperator", "ICmpInst", "FCmpInst", "DetachInst" etc.
StringRef getInstClassName(const Instruction &inst);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_INSTRUCTION_UTILS_H
