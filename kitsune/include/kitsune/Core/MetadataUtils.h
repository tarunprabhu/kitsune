//===- MetadataUtils.h - Utilities for LLVM's metadata ---------*- C++ -*--===//
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
#include "kitsune/Support/TypeTraits.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class LLVMContext;
class Metadata;

namespace detail {

template <typename E, unsigned N = 0>
void add(SmallVector<E, N> &container, const E &elem) {
  container.push_back(elem);
}

template <typename E, unsigned N = 0>
void add(SmallSet<E, N> &container, const E &elem) {
  container.insert(elem);
}

} // namespace detail

/// \addtogroup kitsune
/// @{

/// Utilities to construct metadata nodes from C++ values.
/// @{

template <typename T,
          std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>,
                           int> = 0>
Metadata *toMetadata(const T &val, LLVMContext &ctx);

template <typename T, std::enable_if_t<std::is_string_like_v<T>, int> = 0>
Metadata *toMetadata(const T &val, LLVMContext &ctx);

template <int N> Metadata *toMetadata(const char (&s)[N], LLVMContext &ctx) {
  return toMetadata(StringRef(s), ctx);
}

template <typename T, std::enable_if_t<std::is_enum_v<T>, int> = 0>
Metadata *toMetadata(const T &val, LLVMContext &ctx) {
  return ConstantAsMetadata::get(toConstant(val, ctx));
}

template <typename C,
          std::enable_if_t<std::is_iterable_v<C> && !std::is_string_like_v<C>,
                           int> = 0>
Metadata *toMetadata(const C &container, LLVMContext &ctx) {
  SmallVector<Metadata *, 0> mds;
  for (const auto &elem : container)
    mds.push_back(toMetadata(elem, ctx));
  return MDNode::get(ctx, mds);
}

/// @}

/// Utilities to parse C++ values from metadata nodes.
/// @{

template <typename T, std::enable_if_t<std::is_same_v<T, StringRef>, int> = 0>
std::optional<T> fromMetadata(const Metadata *md);

template <typename T,
          std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>,
                           int> = 0>
std::optional<T> fromMetadata(const Metadata *md);

template <typename T, std::enable_if_t<std::is_enum_v<T>, int> = 0>
std::optional<T> fromMetadata(const Metadata *md) {
  if (auto *cmd = dyn_cast<ConstantAsMetadata>(md))
    if (auto *c = dyn_cast<Constant>(cmd->getValue()))
      return fromConstant<T>(*c);
  return std::nullopt;
}

template <typename C,
          std::enable_if_t<std::is_iterable_v<C> && !std::is_string_like_v<C>,
                           int> = 0>
std::optional<C> fromMetadata(const Metadata *md) {
  if (const auto *mdNode = dyn_cast<MDNode>(md)) {
    C res;
    using E = typename C::value_type;
    for (const Metadata *mdElem : mdNode->operands())
      if (std::optional<E> v = fromMetadata<E>(mdElem))
        detail::add(res, *v);
    if (res.size() == mdNode->getNumOperands())
      return res;
  }
  return std::nullopt;
}

/// @}

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_METADATA_UTILS_H
