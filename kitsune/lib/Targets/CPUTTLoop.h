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

#include "kitsune/Core/TTID.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
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

  /// Generate a wrapper for the function \p outlined. \p outlined is obtained
  /// by outlining the body of a tapir loop. \p outlined is assumed to have the
  /// signature:
  ///
  ///     void(i64 %beg, i64 %end, %args...)
  ///
  /// Here %beg, and %end are the range of iterations that the function operates
  /// on. %args... is a variadic list of arguments that consist of the entities
  /// used by the tapir loop from which \p outlined was obtained.
  ///
  /// The wrapper will have the signature
  ///
  ///     void(i64 %beg i64 %end, ptr %args)
  ///
  /// Here, %args is expected to be a struct. Each element of the struct is an
  /// argument to be passed to \p outlined. The body of the generated wrapper
  /// function will be roughly as follows:
  ///
  ///     void @wrapper(i64 %beg, i64 %end, ptr %args) {
  ///     entry:
  ///         %0 = getelementptr %bundleTy, ptr %args, i32 0, i32 0
  ///         %arg0 = load <ty0>, ptr %0
  ///         %1 = getelementptr %bundleTy, ptr %args, i32 0, i32 1
  ///         %arg0 = load <ty0>, ptr %0
  ///         ...
  ///         <body of outlined function>
  ///         br label %exit
  ///
  ///     exit:
  ///         ret void
  ///     }
  ///
  /// Here, %bundleTy is a struct with each element of the struct corresponding
  /// to an argument in %args... expected by \p outlined. Note that \p outlined
  /// is inlined into the generated wrapper function.
  Function *genWrapperFor(Function &outlined);

public:
  virtual ~CPUTTLoopProcessor() = default;

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
