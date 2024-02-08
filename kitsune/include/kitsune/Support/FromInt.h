//===- FromInt.h - Conversions from ints ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions to convert integers to Kitsune-specific types.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SUPPORT_FROM_INT_H
#define KITSUNE_SUPPORT_FROM_INT_H

#include <cstdint>
#include <optional>

namespace llvm {

/// \addtogroup kitsune
/// @{

/// Convert a 64-bit signed integer to the given type. If the conversion could
/// not be performed, return std::nullopt.
template <typename T> std::optional<T> fromInt(int64_t i);

/// @}

} // namespace llvm

#endif // KITSUNE_SUPPORT_FROM_INT_H
