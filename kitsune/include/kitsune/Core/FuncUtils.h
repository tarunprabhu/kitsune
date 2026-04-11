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
class Function;
class LLVMContext;

/// \addtogroup kitsune
/// @{

/// Get the LLVM context from a function. This is useful when generating code
/// from tablegen macros.
LLVMContext &getContext(const Function &f);

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

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_FUNC_UTILS_H
