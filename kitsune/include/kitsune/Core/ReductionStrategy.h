//===- ReductionStrategy.h - The reduction strategy enum --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The reduction strategy enum.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_REDUCTION_STRATEGY_H
#define KITSUNE_CORE_REDUCTION_STRATEGY_H

#include "kitsune/Support/FromInt.h"
#include "kitsune/Support/FromString.h"
#include "kitsune/Support/ToString.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace llvm {

class raw_ostream;

/// \addtogroup kitsune
/// @{

/// The various reduction strategies that are available. This is the strategy
/// used to perform the final reduction into the result. Some strategies are
/// only relevant for parallel reduction loops that will be executed on the CPU,
/// others are only relevant for GPU-reductions, while some are relevant for any
/// reduction.
///
/// ----
/// Maintainers must avoid changing the numerical values of the members of this
/// enum since doing so will require updating a number of tests. The values
/// roughly correspond to the bitvector shown here:
///
///    .----------------------------------------.
///    |        Unused        | WS | SH | D | N |
///    '----------------------------------------'
///
/// where
///
///    N   Naive
///    D   Direct
///    SH  Shared Memory (GPU)
///    WS  Warp Shuffle (GPU)
///
/// Here, the value of the GPUWarpShuffleWithSharedMemory strategy is 14 because
/// the strategy can be thought of as a combination of the GPUWarpShuffle (8),
/// GPUSharedMemory (4), and Direct (2) strategies - threads within a warp
/// perform a warp shuffle and store their contribution to the final reduction
/// in shared memory, following which the values are reduced into the final
/// result directly using atomics.
///
/// Not all combination of bits is reasonable. It is not clear if this approach
/// will allow any form of extensibility, but it might. The third LSB is unused
/// in case we ever develop a different strategy for CPU's.
enum class ReductionStrategy : uint32_t {
  /// Let the compiler pick a suitable strategy. This should only be used by the
  /// frontend when first generating an intrinsic. By the time the intrinsic is
  /// to be lowered, any uses of this should have been replaced with a specific
  /// strategy.
  Auto = 0x0,

  /// This is the equivalent of a serial reduction. It is performed directly
  /// into the destination without an atomic. This should only be used when a
  /// serial reduction is to be performed. In practice, with Kitsune, this will
  /// usually be seen in the "base" cases for more sophisticated approaches, for
  /// instance, when using warp shuffles on a GPU.
  Naive = 0x1,

  /// Each thread accumulates its contribution directly into the result variable
  /// using an atomic operation. This is the most naive approach possible, and
  /// is likely to have poor performance on both CPU and GPU. In certain cases,
  /// such as when the destination of a reduction in a loop is not loop
  /// invariant, this is the only strategy that will yield correct results. This
  /// is different from the \ref ThreadLocal strategy since it does not compute
  /// a partial result within a thread - all calls to the kit.reduce intrinsic
  /// are essentially replaced with an atomic operation. This matters in cases
  /// where, for instance, a call to a reduce intrinsic is in a serial loop
  /// that is a child of a tapir loop. In such cases, reducing into a
  /// thread-local temporary may be beneficial, but, with this strategy, that
  /// will not be done.
  Direct = 0x2,

  /// Each thread in a block writes its contribution to the result into a
  /// dedicated location in GPU shared memory. These are then reduced into a
  /// single value for the block. This is accumulated into the result variable
  /// using an atomic operation.
  GPUSharedMemory = 0x6,

  /// Each thread in a warp computes its contribution to the result, which are
  /// the reduced to a single value for the warp using warp shuffles. A single
  /// thread in the warp will accumulate this directly into the result variable
  /// using an atomic operation.
  GPUWarpShuffle = 0xA,

  /// Each thread in a warp computes its contribution to the result, which are
  /// the reduced to a single value for the warp using warp shuffles. A single
  /// thread in the warp writes this to a dedicated location in shared memory.
  /// These are then reduced to a single value for the block. This value is
  /// accumulated into the result variable using an atomic operation.
  GPUWarpShuffleWithSharedMemory = 0xE,
};

/// Check if the reduction strategy is valid for reductions that are computed on
/// a CPU.
bool isCPUStrategy(ReductionStrategy strategy);

/// Check if the reduction strategy is valid for reductions that are computed on
/// a GPU.
bool isGPUStrategy(ReductionStrategy strategy);

/// Stream a ReductionStrategy using LLVM's streams.
raw_ostream &operator<<(raw_ostream &os, const ReductionStrategy &v);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_REDUCTION_STRATEGY_H
