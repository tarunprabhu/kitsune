//===- ConstantUtils.h - Helper functions for constants --------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper functions for constants.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_CONSTANT_UTILS_H
#define KITSUNE_CORE_CONSTANT_UTILS_H

#include "kitsune/Core/Tapir.h"
#include "kitsune/Core/TypeUtils.h"
#include "kitsune/Support/FromInt.h"
#include "kitsune/Support/TypeTraits.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"

namespace llvm {

/// \addtogroup kitsune
/// @{

class ConstantInt;
class GlobalVariable;
class LLVMContext;
class Module;

/// Strip all casts from \p c and return the innermost constant. If there are no
/// casts, return the \p c as is.
Constant *stripCasts(Constant *c);
const Constant *stripCasts(const Constant *c);

/// Create a private string with the given initializer if one with this
/// initializer does not already exist in the module. If one does, return that.
/// This is linear in the number of global variables in the module.
///
/// @param s The string initializer
/// @param m The module in which to create the string
/// @param name If a global variable is to be created, the name to give it.
GlobalVariable *createConstString(StringRef s, Module &m, StringRef name = "");

/// Utilities to convert C++ values to LLVM constants
/// @{

template <typename T, std::enable_if_t<std::is_bool_v<T>, int> = 0>
Constant *toConstant(const T &val, LLVMContext &ctx);

template <typename T, std::enable_if_t<std::is_integer_v<T>, int> = 0>
Constant *toConstant(const T &val, LLVMContext &ctx);

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
Constant *toConstant(const T &val, LLVMContext &ctx);

template <typename T, std::enable_if_t<std::is_same_v<T, StringRef> ||
                                           std::is_same_v<T, StringLiteral> ||
                                           std::is_same_v<T, std::string>,
                                       int> = 0>
Constant *toConstant(const T &val, LLVMContext &ctx);

template <int N> Constant *toConstant(const char (&s)[N], LLVMContext &ctx) {
  return toConstant(StringRef(s), ctx);
}

template <typename T, std::enable_if_t<std::is_enum_v<T>, int> = 0>
Constant *toConstant(const T &val, LLVMContext &ctx) {
  return ConstantInt::get(getLLVMTypeFor<int32_t>(ctx), int32_t(val));
}

/// @}

/// Utilities to convert LLVM Constant's to C++ values.
/// @{

template <typename T, std::enable_if_t<std::is_bool_v<T>, int> = 0>
std::optional<T> fromConstant(const Constant &c);

template <typename T, std::enable_if_t<std::is_integer_v<T>, int> = 0>
std::optional<T> fromConstant(const Constant &c);

template <typename T, std::enable_if_t<std::is_float_v<T>, int> = 0>
std::optional<T> fromConstant(const Constant &c);

template <typename T, std::enable_if_t<std::is_double_v<T>, int> = 0>
std::optional<T> fromConstant(const Constant &c);

template <typename T, std::enable_if_t<std::is_same_v<T, StringRef>, int> = 0>
std::optional<T> fromConstant(const Constant &c);

template <typename T, std::enable_if_t<std::is_enum_v<T>, int> = 0>
std::optional<T> fromConstant(const Constant &c) {
  // We assume that all enums are saved as 32-bit integers. This is generally
  // true for the enums that we use in Kitsune right now. If this changes, this
  // could get ugly.
  if (const auto *cint = dyn_cast<ConstantInt>(&c))
    if (cint->getBitWidth() == 32)
      return fromInt<T>(cint->getLimitedValue());
  return std::nullopt;
}

/// @}

/// Get a constant zero for the given type.
Constant *getZero(Type *type);

/// Get a constant one for the given type.
Constant *getOne(Type *type);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_CONSTANT_UTILS_H
