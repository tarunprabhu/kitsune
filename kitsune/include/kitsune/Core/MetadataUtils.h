//===- MetadataUtils.h - Helper functions for LLVM's metadata --*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM's metadata.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_METADATA_UTILS_H
#define KITSUNE_CORE_METADATA_UTILS_H

#include "kitsune/Core/ConstantUtils.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class LLVMContext;
class Metadata;

/// \addtogroup kitsune
/// @{

/// Utilities to construct metadata nodes from C++ values.
/// @{

template <typename T, std::enable_if_t<std::is_same_v<T, StringRef>, int> = 0>
Metadata *toMetadata(T val, LLVMContext &ctx) {
  return MDString::get(ctx, val);
}

template <typename T,
          std::enable_if_t<std::is_enum_v<T> || std::is_integral_v<T> ||
                               std::is_floating_point_v<T>,
                           int> = 0>
Metadata *toMetadata(T val, LLVMContext &ctx) {
  return ConstantAsMetadata::get(toConstant(val, ctx));
}

/// @}

/// Utilities to parse C++ values from metadata nodes.
/// @{

template <typename T, std::enable_if_t<std::is_same_v<T, StringRef>, int> = 0>
std::optional<T> fromMetadata(const Metadata *md) {
  if (auto *mdString = dyn_cast<MDString>(md))
    return mdString->getString();
  return std::nullopt;
}

template <typename T,
          std::enable_if_t<std::is_enum_v<T> || std::is_integral_v<T> ||
                               std::is_floating_point_v<T>,
                           int> = 0>
std::optional<T> fromMetadata(const Metadata *md) {
  if (auto *cmd = dyn_cast<ConstantAsMetadata>(md))
    if (auto *c = dyn_cast<Constant>(cmd->getValue()))
      return fromConstant<T>(*c);
  return std::nullopt;
}

/// @}

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_METADATA_UTILS_H
