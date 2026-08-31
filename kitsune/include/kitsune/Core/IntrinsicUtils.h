//===- IntrinsicUtils.h - Utilities for Kitsune's intrinsics ---*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for Kitsune's intrinsics
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_INTRINSIC_UTILS_H
#define KITSUNE_CORE_INTRINSIC_UTILS_H

#include "kitsune/Core/TTID.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Intrinsics.h"

namespace llvm {

/// \addtogroup kitsune
/// @{

class CallBase;
class Value;

/// How a Kitsune-specific intrinsic is to be lowered. These are used by the
/// intrinsic lowering passes.
enum class KitIntrLowerMode {
  /// The lowering mode is unspecified.
  Unspecified = 0,

  /// The intrinsic is not lowered. The intrinsic lowering passes will delete
  /// calls to this intrinsic.
  Delete,

  /// The intrinsic maps to a function from Kitsune's runtime. The exact
  /// function that it maps to will depend on the tapir target ID and must be
  /// specified in the RTSpec field of the intrinsic definition.
  Runtime,

  /// The intrinsic requires custom handling.
  Custom,
};

/// Get the lowering mode for the Kitsune-specific intrinsic \p id.
KitIntrLowerMode getKitIntrLowerMode(Intrinsic::ID id);

/// Check if the intrinsic is a Kitsune intrinsic.
bool isKitIntrinsic(Intrinsic::ID id);

/// Check if the given Kitsune intrinsic is asynchronous i.e. when the intrinsic
/// returns, the operation that it is intended to perform may not have completed
/// - it may only have been started. It is an error to call this function with
/// an intrinsic that is not a Kitsune intrinsic.
bool isKitIntrinsicAsync(Intrinsic::ID id);

/// Check if the give Kitsune intrinsic is blocking i.e. it only returns when
/// the operation that it is intended to perform has completed. It is an error
/// to call this function with an intrinsic that is not a Kitsune intrinsic.
bool isKitIntrinsicBlocking(Intrinsic::ID id);

/// Check if the given Kitsune intrinsic is CPU-centric i.e. it only applies to
/// the tapir targets that generate multi-threaded CPU code. Note that the
/// 'serial' tapir target is not considered to be CPU-centric since it does not
/// generate parallel code. This will return false for intrinsics that apply to
/// both CPU-centric and GPU-centric tapir targets as well as those that are
/// primarily related to the runtime.
bool isKitIntrinsicCPU(Intrinsic::ID id);

/// Check if the given Kitsune intrinsic is GPU-centric i.e. it only applies to
/// the tapir targets that generate GPU code. This will return false for
/// intrinsics that apply to both CPU-centric and GPU-centric tapir targets as
/// well as those that are primarily related to the runtime.
bool isKitIntrinsicGPU(Intrinsic::ID id);

/// Get the stream from the given call to Kitsune's kernel launch intrinsic.
Value *getStreamFromLaunch(const CallBase &call);

/// Get the TTID argument from a call. This will return std::nullopt if any of
/// the following is not true:
///
///   - The call is not to a Kitsune-specific intrinsic.
///   - The first argument to the call is not an immediate constant integer
///     whose value is not a valid integer representation of a TTID.
///
std::optional<TTID> getTTIDFromKitIntrCall(const CallBase &call);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_INTRINSIC_UTILS_H
