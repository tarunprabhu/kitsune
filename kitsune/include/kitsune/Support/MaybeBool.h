//===- MaybeBool.h - Value that may be a boolean or unset ------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Enumeration for a value that is either a boolean or unset.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SUPPORT_MAYBE_BOOL_H
#define KITSUNE_SUPPORT_MAYBE_BOOL_H

#include <cstdint>

namespace llvm {

/// \ingroup kitsune
/// An enumeration that may be set to a boolean value or unset. This is intended
/// to be used in conjunction with command-line flags - where the presence of
/// the flag usually indicates true. In some cases, it is helpful to distinguish
/// between "false because the flag was not provided", or "false because the
/// negation of the flag was provided". For some flag `X`, the table below
/// is an example of how this type could be used by, say, the driver.
///
///     Command line contains     |  Value of `X` in driver
///     --------------------------|------------------------
///     `-fX`                     |  MaybeBool::On
///     `-fno-X`                  |  MaybeBool::Off
///     <neither -fX, nor -fno-X> |  MaybeBool::Any
///
/// The raw values of this boolean are intentionally chosen to be 0, 1 and 3.
/// We wanted the set value of false to be zero because "false-y" values are
/// generally set to zero. The "true" value is set to 1, though it could, in
/// principle, be any non-zero value. This feels more "natural". The unset value
/// is set to 3 because it implies that both "bits" in this value are set,
/// something that should never happen with this type.
enum class MaybeBool : uint8_t {
  Off = 0, ///< The value is set to false
  On = 1,  ///< The value is set to true
  Any = 3  ///< The value is unset
};

} // namespace llvm

#endif // KITSUNE_SUPPORT_MAYBE_BOOL_H
