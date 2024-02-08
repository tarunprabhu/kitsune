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

class BasicBlock;
class BinaryOperator;
class Instruction;
class LLVMContext;
class Module;
class Value;

/// \addtogroup kitsune
/// @{

/// Get the LLVM context from an instruction. This is useful when generating
/// code from tablegen macros.
LLVMContext &getContext(const Instruction &inst);

/// Get the module containing the instruction, or nullptr if the instruction is
/// not contained in a function, or the containing function is not in a module.
Module *getModule(Instruction &inst);
const Module *getModule(const Instruction &inst);

/// Get the name of an instruction. If the global is unnamed, a string of the
/// form `%<N>` will be returned. This is how the instruction would appear in
/// human-readable LLVM-IR.
std::string getName(const Instruction &inst);

/// Get the class name of the given instruction. This will be a string literal
/// representing the C++ name of the concrete class of the instruction. For
/// instance, "BinaryOperator", "ICmpInst", "FCmpInst", "DetachInst" etc.
StringRef getInstClassName(const Instruction &inst);

/// Is the instruction a call to the llvm.syncregion.start() intrinsic.
bool isCallSyncRegionStart(const Instruction &inst);

/// Replace all operands in the instruction \p inst that are not equal to
/// \p match with \p v. When matching the operands, this will only check for the
/// exact same pointer value - no equivalence checks will be performed. Return
/// true if at least one operand was replaced, false otherwise. The type of \p v
/// must be the same as the type of any operand being replaced. A type mismatch
/// will result in a catastrophic error.
///
/// This is most useful in instructions with two operands where one of the
/// operands is to be replaced.
bool replaceNonMatchingOperands(Instruction &inst, Value *match, Value *v);

/// Replace all operands in the instruction \p inst that are equal to \p match
/// with \p newOp. When matching the operands, this will only check for the
/// exact same pointer value - no equivalence checks will be performed. Return
/// true if at least one operand was replaced, false otherwise. The type of \p v
/// must be the same as the type of any operand being replaced. A type mismatch
/// will result in a catastrophic error.
///
/// This is most useful in instructions with two operands where one of the
/// operands is to be replaced.
bool replaceMatchingOperands(Instruction &inst, Value *match, Value *v);

/// Get the operand in the instruction \p inst that is not equal to \p match.
/// If neither operand matches \p match, return \p nullptr.
Value *getNonMatchingOperand(BinaryOperator &binOp, Value *match);

/// Get the successor of the instruction that does not match the given basic
/// block. If the instruction does not have exactly two successors, returns
/// nullptr.
BasicBlock *getNonMatchingSuccessor(Instruction &inst, BasicBlock *bb);

/// Is the given instruction a conditional branch.
bool isCondBr(const Instruction &inst);

/// Is the given instruction an unconditional branch.
bool isUncondBr(const Instruction &inst);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_INST_UTILS_H
