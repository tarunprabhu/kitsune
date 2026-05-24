//===- ValueUtils.h - Utilities for LLVM Value's ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM values.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_VALUE_UTILS_H
#define KITSUNE_CORE_VALUE_UTILS_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

class Module;
class Type;
class Value;

/// \addtogroup kitsune
/// @{

/// Get the module in which the given LLVM Value is contained. \p v must be an
/// Argument, BasicBlock, GlobalValue, or Instruction. In all other cases, this
/// will always return nullptr.
Module *getModule(Value &v);
const Module *getModule(const Value &v);

/// Get the name of an LLVM Value. If the value does not have a name, a string
/// that matches how the value would be rendered in LLVM-IR is returned. If the
/// value is a function or global variable, this will be of the form `@<N>`
/// where <N> is a non-negative integer. If the value is an instruction, this
/// will be of the form `%<N>`.
///
/// \p v must be an Argument, BasicBlock, GlobalValue, or Instruction. It is an
/// error to pass any other value, such a Constant to this function.
std::string getName(const Value &v);

/// Check that the given value is a constant `false`.
bool isFalse(const Value *v);

/// Check that the given value is a constant `true`.
bool isTrue(const Value *v);

/// Check that the given value is a constant 0 of integer or floating point
/// type.
bool isZero(const Value *v);

/// Check that the given value is a constant 0 of the given type.
bool isZero(const Value *v, Type *ty);

/// Check that the given value is a constant 1 of integer type.
bool isIntOne(const Value *v);

/// Check that the given value is a constant 1 of the given type.
bool isIntOne(const Value *v, Type *ty);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_VALUE_UTILS_H
