//===- Sections.h - Kitsune-specific object-file sections ------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for object-file sections used by Kitsune for various purposes.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_OBJECT_SECTIONS_H
#define KITSUNE_OBJECT_SECTIONS_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

/// \addtogroup kitsune
/// @{

namespace object {

/// The name of the section that contains embedded bitcode.
static constexpr StringRef kitSectEmbBC = ".kit.mbc";

/// The name of the section that contains the runtime initialization options.
static constexpr StringRef kitSectRTInitOpts = ".kit.rtiopt";

} // namespace object

/// @}

} // namespace llvm

#endif // KITSUNE_OBJECT_SECTIONS_H
