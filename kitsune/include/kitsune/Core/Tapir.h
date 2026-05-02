//===- Tapir.h - Core Kitsune types and enums ------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file enumerates the available Tapir lowering targets and other types
// that are shared between the front and middle-ends.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_TAPIR_H
#define KITSUNE_CORE_TAPIR_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace llvm {

/// \addtogroup kitsune
/// @{

/// The identifiers for the known tapir targets.
///
/// These are some useful constraints that it would be useful to maintain.
/// The Nolo tapir target does not perform any lowering - instead, the tapir
/// instructions are retained in the tapir loop even after it has been
/// "lowered". This should always have an integer value of 0. The serial tapir
/// target simply lowers the tapir loop to a serial loop. This should have an
/// integer value of 1 because it "makes sense" in this context - a serial loop
/// can be thought of as a special case of a parallel loop where only a single
/// iteration is executing at a time.
///
/// NOTES:
///
///   1. The values of these enums should not be changed unless absolutely
///      necessary. A number of tests hardcode these values since the integer
///      values appear in metadata nodes in LLVM-IR (and, in the future, also in
///      MLIR).
///
///   2. The values are intentionally powers of two - essentially setting a
///      single bit in a bit-vector. It may allow us to efficiently encode
///      multiple tapir targets for use in a single invocation of the compiler.
///      That's the idea for now at least, but it's not clear that we will ever
///      use this feature. 32 tapir targets ought to be more than enough for
///      everyone.
///
///   3. The underlying type is a 32-bit integer because that is what is used
///      when lowering this to LLVM IR for use in loop metadata.
///
enum class TTID : uint32_t {
  /// Pseudo tapir target that does not lower tapir instructions. This is
  /// primarily useful to generate LLVM IR containing tapir instructions.
  Nolo = 0x0,

  /// Lower to a straightforward serial implementation.
  Serial = 0x1,

  /// Lower to run on NVIDIA GPU's.
  Cuda = 0x2,

  /// Lower to run on AMD GPU's.
  Hip = 0x4,

  /// Lower to use the OpenCilk runtime.
  OpenCilk = 0x8,

  // Lower to a JIT-enabled, GPU-agnostic runtime. The original implementation
  // of this is largely obsolete, but we may add a new JIT-enabled runtime in
  // the future. Until then, we reserve the actual ID.
  // GPUABI = 0x10,

  /// Lower to use the Qthreads runtime.
  Qthreads = 0x20,

  /// Lower to use Legion's Realm runtime.
  /// FIXME: This is currently disabled and needs to be updated before it can be
  /// re-enabled.
  Realm = 0x40,

  /// Lower using a generic tapir target that uses bitcode files containing the
  /// bulk of the code used to lower a tapir loop (or other construct).
  /// FIXME: This has not been fully implemented or tested.
  Lambda = 0x80,

  /// Lowering to the OpenMP task ABI.
  /// FIXME: This has not been fully implemented or tested.
  OMPTask = 0x100,

  /// Lower to use OpenMP's runtime. This will use libomp that is built together
  /// with Kitsune. Using other OpenMP libraries is not (and likely will never
  /// be) supported.
  OpenMP = 0x200,

  /// Lowering to use POSIX threads (pthreads). On POSIX platforms, these are
  /// guaranteed to be available.
  Pthreads = 0x400,

  /// Lower using a tapir target that is loaded from tapir target plugin. The
  /// plugin is a dynamic shared object.
  Custom = 0x800,
};

/// The default primary tapir target. This is present simply to reiterate the
/// fact that a primary tapir target may not be provided. Currently we require
/// a primary tapir target in order to enable tapir-related lowering and
/// code generation, both in the frontend and directly in the middle-end.
static constexpr std::optional<llvm::TTID> defaultTapirTarget = std::nullopt;

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

/// The default grain size. This is set to 0 instead of 1 because the value of
/// zero also doubles up as the absence of an explicitly specified grain size.
static constexpr unsigned defaultTapirGrainSize = 0;

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_TAPIR_H
