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
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"

namespace llvm {

/// \addtogroup kitsune
/// @{

class ConstantInt;
class GlobalVariable;
class LLVMContext;
class Module;

/// Create a TTID from the given constant.
std::optional<TTID> createTTIDFrom(const ConstantInt &);

/// Generate a ConstantInt for use in Kitsune-specific intrinsics that take a
/// tapir target id as an argument.
ConstantInt *createConstInt(TTID tt, LLVMContext &ctxt);

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

template <typename T, std::enable_if_t<std::is_enum_v<T>, int> = 0>
Constant *toConstant(LLVMContext &ctx, T val) {
  return ConstantInt::get(getLLVMTypeFor<int32_t>(ctx), int32_t(val));
}

template <typename T, std::enable_if_t<std::is_same_v<T, StringRef>, int> = 0>
Constant *toConstant(LLVMContext &ctx, T val) {
  return ConstantDataArray::getString(ctx, val, /*AddNull=*/false);
}

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
Constant *toConstant(LLVMContext &ctx, T val) {
  return ConstantInt::get(getLLVMTypeFor<T>(ctx), val);
}

/// @}

/// Utilities to convert LLVM Constant's to C++ values.
/// @{
template <typename T, std::enable_if_t<std::is_same_v<T, StringRef>, int> = 0>
std::optional<StringRef> fromConstant(const Constant &c) {
  if (const auto *cda = dyn_cast<ConstantDataArray>(&c)) {
    if (cda->isString())
      return cda->getAsString();
    else if (cda->isCString())
      return cda->getAsCString();
  }
  return std::nullopt;
}

template <typename T, std::enable_if_t<std::is_same_v<T, TTID>, int> = 0>
std::optional<T> fromConstant(const Constant &c) {
  if (const auto *cint = dyn_cast<ConstantInt>(&c))
    return createTTIDFrom(cint->getLimitedValue());
  return std::nullopt;
}

template <typename T,
          std::enable_if_t<std::is_same_v<T, TapirSpawnStrategy>, int> = 0>
std::optional<T> fromConstant(const Constant &c) {
  if (const auto *cint = dyn_cast<ConstantInt>(&c))
    return createTapirSpawnStrategyFrom(cint->getLimitedValue());
  return std::nullopt;
}

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
std::optional<T> fromConstant(const Constant &c) {
  if (const auto *cint = dyn_cast<ConstantInt>(&c))
    return cint->getLimitedValue();
  return std::nullopt;
}
/// @}

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_CONSTANT_UTILS_H
