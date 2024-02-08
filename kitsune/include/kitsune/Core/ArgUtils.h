//===- ArgUtils.h - Utilities for LLVM function arguments ------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM function Argument's.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_ARG_UTILS_H
#define KITSUNE_CORE_ARG_UTILS_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

class Argument;
class LLVMContext;
class Module;

/// \addtogroup kitsune
/// @{

/// Get the module containing the function for which \p a is an argument, or
/// nullptr, if the function is not in a module.
Module *getModule(Argument &a);
const Module *getModule(const Argument &a);

/// Get the LLVM context from a function. This is useful when generating code
/// from tablegen macros. This requires the argument to have a parent function.
LLVMContext &getContext(const Argument &a);

/// Get the name of a function argument. If the argument is unnamed, a string of
/// the form `%<N>` will be returned. This is how the argument name would appear
/// in human-readable LLVM-IR.
std::string getName(const Argument &a);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_ARG_UTILS_H
