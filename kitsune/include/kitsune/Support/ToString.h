//===- ToString.h - Convert Kitsune-specific types to string ---*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to convert Kitsune-specific types to string. This could be used to
// convert other types to string as well.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SUPPORT_TO_STRING_H
#define KITSUNE_SUPPORT_TO_STRING_H

#include "kitsune/Core/OptznLevel.h"
#include "kitsune/Core/Tapir.h"

namespace llvm {

/// \addtogroup kitsune
/// @{

std::string toString(const TTID &tt);
std::string toString(const MaybeBool &);
std::string toString(const OptznLevel &);

/// @}

} // namespace llvm

#endif // KITSUNE_SUPPORT_TO_STRING_H
