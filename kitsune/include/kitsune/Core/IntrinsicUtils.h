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
