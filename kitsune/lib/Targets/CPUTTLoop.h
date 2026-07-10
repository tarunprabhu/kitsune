//===- CPUTTLoop.h - CPU-centric loop outline processor --------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class for loop outline processors used by the CPU-centric
// threading-focused tapir target such as openmp, pthreads, and qthreads.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TARGETS_CPUTT_LOOP_H
#define KITSUNE_TARGETS_CPUTT_LOOP_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {

/// \ingroup kitsune
/// Base class shared by the loop outline processors of CPU-centric,
/// threading-focused tapir targets that do not use a grainsize argument.
class CPUTTLoopProcessor : public LoopOutlineProcessor {
protected:
  TTID tt;

  // If true, the outline loop call will be replaced with an asynchronous thread
  // launch intrinsic, otherwise, a blocking thread launch intrinsic will be
  // used.
  bool asyncLaunch;

protected:
  CPUTTLoopProcessor(TTID tt, const TTOptions &opts, bool asynLaunch,
                     Module &m);

  /// Insert an asynchronous launch call that will eventually replace \p call.
  /// Consider the call below:
  ///
  ///     call void @func(args ...)
  ///
  /// This will be replaced with something similar to what is shown below.
  ///
  ///     %ctx = call @llvm.kit.async.cpu.threads.launch(%tt, @func, args...)
  ///     call @llvm.kit.cpu.threads.sync(%tt, %ctx)
  ///
  void insertAsyncLaunch(CallBase &call);

  /// Insert a blocking launch call that will eventually replace \p call.
  /// Consider the call below:
  ///
  ///     call void @func(args ...)
  ///
  /// This will be replaced with something similar to what is shown below.
  ///
  ///     call @llvm.kit.cpu.threads.launch(%tt, @func, args...)
  ///
  void insertBlockingLaunch(CallBase &call);

public:
  virtual ~CPUTTLoopProcessor() = default;

  /// Returns an ArgStructMode enum value describing how inputs to the
  /// underlying task of a tapir loop should be passed to the task.
  virtual ArgStructMode getArgStructMode() const override;

  /// Setup the loop-control arguments \p lcArgs and loop-control inputs
  /// \p lcInputs for the Tapir loop \p tl.
  virtual void
  setupLoopControlArgs(TapirLoopInfo *tl, SmallVectorImpl<Value *> &lcArgs,
                       SmallVectorImpl<Value *> &lcInputs) override;

  /// Processes a call to the function into which the tapir loop \p tl has been
  /// outlined.
  virtual void processOutlinedLoopCall(TapirLoopInfo &tl, TaskOutlineInfo &toi,
                                       DominatorTree &dt) override;
};

} // namespace llvm

#endif // KITSUNE_TARGETS_CPUTT_LOOP_H
