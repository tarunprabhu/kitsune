//===- RTID.h - Enum for Kitsune's constituent runtimes ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ID's for Kitsune's constituent runtimes
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SHARED_RTID_H
#define KITSUNE_SHARED_RTID_H

#include <stdint.h>

/// \addtogroup kitsune
/// @{

/// The ID's for the tapir-target specific runtimes and other supporting
/// runtimes that are part of Kitsune's main runtime.
///
/// The values for each of these enumerations is carefully chosen. The runtimes
/// for the tapir targets have exactly the same numerical value as their
/// corresponding llvm::TTID. The llvm::TTID members are each at most 32-bits,
/// so the other supporting runtimes must have values greater than 0xffffffff.
///
/// Note that not all tapir targets have a corresponding runtime. In such cases,
/// the value of the corresponding tapir target must not appear here.
///
/// The value for RT_COMMON *must* always be 0. This is a special value since it
/// is always "implicitly enabled".
enum KitRTID : uint64_t {
  RT_COMMON = 0x0ULL,        ///< Common parts of the runtime.
  RT_CUDA = 0x2ULL,          ///< Runtime for the cuda tapir target
  RT_HIP = 0x4ULL,           ///< Runtime for the hip tapir target
  RT_OPENCILK = 0x8ULL,      ///< Runtime for the opencilk tapir target
  RT_QTHREADS = 0x20ULL,     ///< Runtime for the qthreads tapir target
  RT_OPENMP = 0x200ULL,      ///< Runtime for the openmp tapir target
  RT_PTHREADS = 0x400ULL,    ///< Runtime for the pthreads tapir target
  RT_PAPI = 0x100000000ULL,  ///< Support runtime for PAPI
  RT_TIMER = 0x200000000ULL, ///< Support runtime for timers
};

#ifdef __cplusplus

namespace kitrt {

// TODO: We need C++20 before we can import the members of KitRTID into this
// namespace and turn RTID into a scoped enum.
using RTID = KitRTID;

} // namespace kitrt

#endif // __cplusplus

/// @}

#endif // KITSUNE_SHARED_RTID_H
