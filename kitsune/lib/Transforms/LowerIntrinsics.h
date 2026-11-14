//===- LowerIntrinsics.h - Lower Kitsune's intrinsics ----------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private implementation to lower Kitsune's reduce intrinsics. This only
// handles a subset of the intrinsics that need to be lowered in the compiler
// middle-end. Most intrinsics are lowered during codegen.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_TRANSFORMS_LOWER_INTRINSICS_H
#define KITSUNE_LIB_TRANSFORMS_LOWER_INTRINSICS_H

#include <cstdint>

namespace llvm {

class CallInst;
class Function;
class Module;
class TTOptions;

namespace detail {

/// \addtogroup kitsune
/// @{

/// The "categories" of intrinsics to be lowered. Depending on the intrinsic, it
/// may have to be lowered at different points in Kitsune's pipeline. For
/// instance, reduce intrinsics intended to run on a GPU must be lowered only
/// after outlining into a device code module since the lowering will typically
/// generate convergent code that is not allowed in standard tapir loops.
///
/// The kinds here are logical groupings and do not necessarily correspond to
/// particular intrinsics, or even groups of them. For instance, the \ref
/// GPUReduce intrinsic corresponds to the @llvm.kit.reduce intrinsic, but where
/// the TTID argument in a call corresponds to a GPU-centric tapir target. On
/// the other hand, \ref GPUIndex refers to a set of intrinsics. On the other,
/// \ref CPUNumThreads refers to just the @llvm.kit.cpu.num.threads intrinsic.
///
/// The members of this enumeration are flags - the integer representation of
/// each member must have a single set bit. These can be OR'ed together to form
/// a mask that will be used by the \ref lowerIntrinsics functions.
///
enum class IntrinsicKind : uint64_t {
  /// Kind for an intrinsic that does not fit into any known category.
  Unknown = 0x0,

  /// The @llvm.kit.cpu.num.threads intrinsic called with the 'serial' tapir
  /// target.
  SerialNumThreads = 0x1,

  /// This group consists of Kitsune's reduce intrinsic where the TTID
  /// argument of the call corresponds to a CPU-centric tapir target.
  CPUReduce = 0x2,

  /// This group consists of Kitsune's reduce intrinsic where the TTID
  /// argument of the call corresponds to a GPU-centric tapir target.
  GPUReduce = 0x4,

  /// This group consists of "index" intrinsics that correspond to the indices
  /// of thread groups and blocks on GPU's. These also include intrinsics that
  /// can be used obtain the sizes of thread blocks and grids.
  GPUIndex = 0x8,

  /// This group consists of intrinsics that will be lowered to GPU operations
  /// on warps.
  GPUWarp = 0x10,
};

/// Type of a the bitmask obtained by OR'ing members of \ref IntrinsicKind.
using IntrinsicKinds = uint64_t;

inline IntrinsicKinds operator|(IntrinsicKind a, IntrinsicKind b) {
  return IntrinsicKinds(a) | IntrinsicKinds(b);
}

inline IntrinsicKinds operator|(IntrinsicKinds n, IntrinsicKind kind) {
  return n | IntrinsicKinds(kind);
}

/// Implementation class to lower Kitsune's intrinsics in the middle-end.
class LowerIntrinsicsImpl {
protected:
  const TTOptions &tto;

  /// Flags that determine the subset of intrinsics that are lowered by an
  /// instance of this class. This is a mask obtained by OR'ing the members of
  /// the \ref IntrinsicKind enum.
  IntrinsicKinds kindsToLower;

protected:
  bool shouldLower(IntrinsicKind kind) const;

  /// Lower a call to an index intrinsic. These intrinsics correspond to the
  /// threadIdx, blockIdx, blockDim, gridSize, and gridDim builtin objects. The
  /// call will be erased from the function. Always returns true.
  bool lowerGPUIndexIntrinsic(CallInst *call);

  /// Lower a call to an intrinsic that will be lowered to an operation on a GPU
  /// warp. The call will be erased from the function. Always returns true.
  bool lowerGPUWarpIntrinsic(CallInst *call);

  /// Lower a call to Kitsune's kit.reduce intrinsic. The call will be erased
  /// from the function. Always returns true.
  bool lowerReduceIntrinsic(CallInst *call);

  /// Lower a call to the given intrinsic. The original call will be erased from
  /// the parent after lowering. Always returns true.
  bool lowerIntrinsic(CallInst *call);

public:
  /// Construct an instance of the intrinsic lowering helper class. \p flags is
  /// a bit mask where each bit corresponds to an intrinsic kind to be lowered.
  /// This is obtained by OR'ing members of \ref IntrinsicKind.
  LowerIntrinsicsImpl(const TTOptions &tto, IntrinsicKinds kindsToLower)
      : tto(tto), kindsToLower(kindsToLower) {}

  /// Lower calls to Kitsune's intrinsic in the given function. Lowering an
  /// intrinsic may result in calls to other intrinsics being created. This
  /// function will iterate until there are no intrinsics to be lowered. Returns
  /// true if at least one intrinsic was lowered, false otherwise.
  bool runOn(Function &f);

  /// Lower calls to Kitsune's intrinsic in the given module. Returns true if at
  /// least one intrinsic was lowered, false otherwise.
  bool runOn(Module &m);
};

/// @}

} // namespace detail

} // namespace llvm

#endif // KITSUNE_LIB_TRANSFORMS_LOWER_INTRINSICS_H
