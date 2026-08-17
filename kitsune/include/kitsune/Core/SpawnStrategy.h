//===- SpawnStrategy.h - The spawn strategsy enum --------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define the core spawn strategy enum.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_SPAWN_STRATEGY_H
#define KITSUNE_CORE_SPAWN_STRATEGY_H

#include "kitsune/Support/FromInt.h"
#include "kitsune/Support/FromString.h"
#include "kitsune/Support/ToString.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace llvm {

class raw_ostream;

/// \addtogroup kitsune
/// @{

/// The loop spawning strategy. For tapir targets that spawn tasks, this
/// determines how the iteration space of the loop is split up across these
/// tasks. In the case of the DivideAndConquer strategy, for instance, the
/// iteration space is recursively divided up across some number of tasks to be
/// spawned. This also affects the loop outline processor that is used and
/// whether or not the code in a tapir loop is outlined. Some tapir targets will
/// only work correctly with certain strategies.
///
/// NOTES:
///
///   1. The integer values of the spawn strategies should not be changed
///      unless absolutely necessary. These integer values are used in the loop
///      metadata in the LLVM IR and are, therefore, hardcoded into several
///      tests.
///
///   2. The integer value of 0 is intentionally unused and should not be used
///      for any spawn strategies since it could also be used to indicate the
///      absence of a strategy.
///
///   3. The value of the Sequential strategy is intentionally 1 since the value
///      is also suggestive of a single task.
///
///   4. The underlying integer type is explicitly declared to be a 32-bit
///      integer since that is what is used when lowering this to LLVM IR for
///      use in loop metadata.
///
enum class TapirSpawnStrategy : uint32_t {
  /// Do not spawn any tasks. Individual tapir targets may choose to outline the
  /// tapir loop.
  Sequential = 1,

  /// Recursively divide the iteration space with a task being spawned for each
  /// leaf of this tree. The leaf will be responsible for 1 or more iterations.
  DivideAndConquer,

  /// Spawn strategy for GPU's. This essentially the same as the sequential
  /// strategy but requires outlining. In the GPU tapir targets that we
  /// currently (as of Oct 2025) support, a custom loop outline
  /// processor is used anyway, so this currently just acts as a marker. But it
  /// may be useful in the future.
  GPU,

  /// This is similar to the sequential strategy but requires outlining. The
  /// difference between this and the GPU strategy is that it is intended for
  /// use by CPU-centric tapir targets. Typically, the targets will split the
  /// iteration space evenly across tasks. These tasks are not intended to
  /// spawn subtasks to further divide the space that they have been assigned.
  Basic,
};

/// The default tapir spawn strategy. This is used because a default is needed
/// by frontends. This reduces the likelihood of accidentally introducing
/// inconsistencies.
static constexpr TapirSpawnStrategy defaultTapirSpawnStrategy =
    TapirSpawnStrategy::Sequential;

/// Stream a TTID using LLVM's streams.
raw_ostream &operator<<(raw_ostream &os, const TapirSpawnStrategy &v);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_SPAWN_STRATEGY_H
