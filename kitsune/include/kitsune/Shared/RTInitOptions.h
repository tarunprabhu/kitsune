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

#include <stdint.h>

/// \addtogroup kitsune
/// @{

/// The ID's for the tapir-target specific runtimes and other supporting
/// runtimes that are part of Kitsune's main runtime.
///
/// The values for each of these enumerations is carefully chosen. The runtimes
/// for the tapir targets have exactly the same numerical value as their
/// corresponding llvm::TTID. The llvm::TTID members are each at most 32-bits,
/// so the othe supporting runtimes must have values greater than 1 << 31.
///
/// Note that not all tapir targets have a corresponding runtime. Even so, a
/// member corresponding to those tapir targets must be present in this enum.
/// These are referred to as "pseudo" runtimes.
enum KitRTID : uint64_t {
  RT_NONE = 0x0ULL,          ///< Only initialize the common runtime
  RT_SERIAL = 0x1ULL,        ///< Runtime for the serial tapir target (pseudo)
  RT_CUDA = 0x2ULL,          ///< Runtime for the cuda tapir target
  RT_HIP = 0x4ULL,           ///< Runtime for the hip tapir target
  RT_OPENCILK = 0x8ULL,      ///< Runtime for the opencilk tapir target
  RT_QTHREADS = 0x20ULL,     ///< Runtime for the qthreads tapir target
  RT_OPENMP = 0x200ULL,      ///< Runtime for the openmp tapir target
  RT_PTHREADS = 0x400ULL,    ///< Runtime for the pthreads tapir target
  RT_PAPI = 0x100000000ULL,  ///< Support runtime for PAPI
  RT_TIMER = 0x200000000ULL, ///< Support runtime for timers
};

/// Initialization options for Kitsune's runtime. An instance of this will be
/// created by Kitsune when it generates a global constructor that will
/// initialize Kitsune's runtime.
struct KitRTInitOptions {
  /// A bitvector of the ID's of the tapir targets whose supporting runtimes are
  /// to be enabled. This is 64-bits because the RTID enum is guaranteed to be
  /// 64-bits. Each bit in this bitvector represents an RTID. If it is set, then
  /// the supporting runtime should be enabled.
  ///
  /// An instance of this object will be created by the compiler. It will set a
  /// bit for all tapir targets used by the code being compiled - even if a
  /// corresponding runtime does not exist for the tapir target (this is the
  /// case, for instance, with the serial tapir target).
  uint64_t rts;
};

#ifdef __cplusplus

namespace kitrt {

using RTID = KitRTID;
using InitOptions = KitRTInitOptions;

} // namespace kitrt

#endif // __cplusplus

/// @}

#endif // KITSUNE_SHARED_RTCONFIG_H
