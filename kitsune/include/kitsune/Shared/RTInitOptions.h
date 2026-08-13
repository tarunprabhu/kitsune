//==- RTInitOptions.h - Options to initialize Kitsune's runtime -*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Initialization options for Kitsune's runtime.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SHARED_RT_INIT_OPTIONS_H
#define KITSUNE_SHARED_RT_INIT_OPTIONS_H

#include "kitsune/Shared/RTID.h"

#include <stdint.h>

/// \addtogroup kitsune
/// @{

/// Initialization options for Kitsune's runtime. An instance of this will be
/// created by Kitsune when it generates a global constructor that will
/// initialize Kitsune's runtime.
typedef struct KitRTInitOptions {
  /// A bitvector of the ID's of the tapir targets whose supporting runtimes are
  /// to be enabled. This is 64-bits because the RTID enum is guaranteed to be
  /// 64-bits. Each bit in this bitvector represents an RTID. If it is set, then
  /// the supporting runtime should be enabled. An instance of this will be
  /// created by the compiler and passed to the runtime.
  uint64_t rts;
} KitRTInitOptions;

#ifdef __cplusplus

namespace kitrt {

using InitOptions = KitRTInitOptions;

} // namespace kitrt

#endif // __cplusplus

/// @}

#endif // KITSUNE_SHARED_RTCONFIG_H
