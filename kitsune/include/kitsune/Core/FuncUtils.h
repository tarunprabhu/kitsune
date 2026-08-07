//===- FuncUtils.h - Utilities for LLVM functions --------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM Function's.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_FUNC_UTILS_H
#define KITSUNE_CORE_FUNC_UTILS_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

class Argument;
class BasicBlock;
class Function;
class LLVMContext;
class Module;

/// \addtogroup kitsune
/// @{

/// Get the LLVM context from a function. This is useful when generating code
/// from tablegen macros.
LLVMContext &getContext(const Function &f);

/// Get the module containing a function, or nullptr, if the function is not in
/// a module.
Module *getModule(Function &f);
const Module *getModule(const Function &f);

/// Get the name of a function. If the function is unnamed, a string of the form
/// `@<N>` will be returned. This is how the function name would appear in
/// human-readable LLVM-IR.
std::string getName(const Function &f);

/// Copy function attributes and other properties from the function \p src to
/// the function \p dst. This will *NOT* copy attributes on function arguments.
/// In order to copy those, use the copyAttributesFrom() method on a Function.
///
///  - calling convention
///  - garbage collection algorithm
///  - personality function
///  - prefix data
///  - prolog data
///
void copyAttrs(Function &dst, const Function &src);

/// Copy attributes from the argument \p src to the argument \p dst.
void copyAttrs(Argument &dst, const Argument &src);

/// Get the basic block with the name \p name in function \f, or nullptr if such
/// a block does not exist.
BasicBlock *getBlockNamed(StringRef name, Function &f);

/// Sort the basic blocks in the function so they are in some "reasonable"
/// order, usually something that resembles "program order". In most cases,
/// this is just reverse postorder, but it may be some other hybrid ordering.
///
/// This is mainly useful when printing the function during testing/debugging.
/// A particular case is when Tapir's loop outliner runs. Since it treats all
/// nested tapir loops as subtasks of the outermost loop, it moves basic blocks
/// based on the spindles to which they belong. This complicates testing for
/// the non-opencilk tapir targets since they would prefer the blocks to be in
/// the order in which they were in the original tapir loop.
///
/// Returns true if sorting was attempted. This will return true even if the
/// basic blocks were already sorted. Essentially, this will only return false
/// if \p f does not contain any basic blocks.
bool sortBasicBlocks(Function &f);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_FUNC_UTILS_H
