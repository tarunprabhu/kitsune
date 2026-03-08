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

/// \addtogroup kitsune
/// @{

class LLVMContext;
class Metadata;

/// Construct an MDNode for tapir loop metadata.
///
/// FIXME: Remove this in favor of the loop attribute metadata mechanisms that
/// have been introduced. This is currently only used in clang - and that part
/// of the code is fairly messy already.
///
/// \tparam DstTy The C++ type of the value in the metadata node. This should be
/// a C++ type, e.g. uint32_t if the \tparam SrcType should be appear as a
/// 32-bit integer in the constructed metadata node
/// \tparam SrcTy The type of the value to be serialized in the loop metadata
/// node. This is the type of \p val.
/// \param ctx The LLVM context object.
/// \param name A string to use as the first operand of the MDNode that will be
/// constructed. This will typically be of the form tapir.loop.*
/// \param val The value to add as the second operand of the MDNode to be
/// constructed. This is the actual value of the tapir loop metadata whose name
/// is \p name.
template <typename DstTy, typename SrcTy>
Metadata *makeTapirLoopMetadata(LLVMContext &ctx, StringRef name,
                                const SrcTy &val);

/// Utilities to construct metadata nodes from C++ values.
/// @{

template <typename T, std::enable_if_t<std::is_same_v<T, StringRef>, int> = 0>
Metadata *toMetadata(T val, LLVMContext &ctx) {
  return MDString::get(ctx, val);
}

template <typename T,
          std::enable_if_t<std::is_enum_v<T> || std::is_integral_v<T>, int> = 0>
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
          std::enable_if_t<std::is_enum_v<T> || std::is_integral_v<T>, int> = 0>
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
