//===- TapirLoopNestAnalysis.h - Analyze nests of tapir loops --*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tools to deal with nests of tapir loops. The name of this file is
// intentionally similar to the llvm/Analysis/LoopNestAnalysis. In the latter
// case, however, there is an actual loop analysis pass that will return a
// loop nest object. There is no such analysis pass for tapir loops.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_ANALYSIS_TAPIR_LOOP_NEST_ANALYSIS_H
#define KITSUNE_ANALYSIS_TAPIR_LOOP_NEST_ANALYSIS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopNestAnalysis.h"

namespace llvm {

class LoopInfo;
class TaskInfo;

/// An object that wraps a tapir loop nest. A tapir loop nest is a loop nest
/// where the root is a tapir loop. Subloops may or may not be tapir loops.
/// All the loop nests shown next are tapir loop nests (we assume that the
/// forall statement is lowered to a tapir loop).
///
/// \code
///     forall        forall      forall      forall
///       forall        ...         for         for
///         forall      forall        for         forall
///           ...         ...           ...         ...
/// \endcode
///
/// However, the loop nests shown next are not tapir loop nests.
///
/// \code
///     for
///       forall
///         ...
/// \endcode
///
/// Note that the names of several methods provided by this object are
/// identical to those provided by LLVM's LoopNest object. However, the values
/// returned here may be different from what LoopNest might return. The two
/// objects are *NOT* intended to be interchangeable.
///
class TapirLoopNest {
private:
  LoopNest nest;
  LoopVectorTy perfectTapirLoops;

private:
  TapirLoopNest(Loop &loop, TaskInfo &ti, ScalarEvolution &se);

  /// When constructing the loop nest, check some basic properties of an "outer"
  /// loop. This is relative to some other "inner" loop. It is not necessarily
  /// the root of the tapir loop nest. Return false if at least one property
  /// does not hold, true otherwise.
  bool sanityCheckOuterLoop(const Loop &loop, ScalarEvolution &se) const;

  /// When constructing the loop nest, check some basic properties of an "inner"
  /// loop. Return false if at least one property does not hold, true otherwise.
  bool sanityCheckInnerLoop(const Loop &loop, ScalarEvolution &se) const;

public:
  /// Get the maximum perfect nesting depth of tapir loops in the nest.
  /// For example, given the loop nest:
  ///
  /// \code
  ///   forall(i)     // loop at level 1
  ///     forall(j)   // loop at level 2
  ///       <code>
  ///       forall(k) // loop at level 3
  /// \endcode
  ///
  /// getMaxPerfectDepth() would return 2. This is what one would expect
  /// since the loop at level 3 is clearly not perfectly nested.
  ///
  /// However, given the loop nest below, getMaxPerfectDepth() would also return
  /// 2.
  ///
  /// \code
  ///   forall(i)     // loop at level 1
  ///     forall(j)   // loop at level 2
  ///       for(k)    // loop at level 3
  /// \endcode
  ///
  /// In this case, although the loop at level 3 is perfectly nested relative
  /// to the outer loop at level 2, it is not a tapir loop.
  ///
  /// For the same reason, given the example below, getMaxPerfectDepth would
  /// return 1.
  ///
  /// \code
  ///   forall(i)     // loop at level 1
  ///     for(j)      // loop at level 2
  ///       forall(k) // loop at level 3
  /// \endcode
  ///
  unsigned getMaxPerfectDepth() const { return perfectTapirLoops.size(); }

  /// Return true if the loop nest is a perfect nest of tapir loops. This
  /// implies that both getNestDepth() and getMaxPerfectDepth() are equal.
  unsigned isPerfect() const { return getMaxPerfectDepth() == getNestDepth(); }

  /// Get only the perfectly nested tapir loops in the nest.
  ArrayRef<Loop *> getPerfectTapirLoops() const { return perfectTapirLoops; }

  /// Get the loop nest depth i.e. the depth of the 'deepest' loop. For example,
  /// given the loop nest:
  ///
  /// \code
  ///   forall(i)         // loop at level 1 and Root of the nest
  ///     for(j1)         // loop at level 2
  ///       forall(k)     // loop at level 3
  ///     forall(j2)      // loop at level 2
  /// \endcode
  ///
  /// getNestDepth() would return 3. Note that this does not distinguish between
  /// tapir loops and non-tapir loops when determining the nest depth.
  ///
  unsigned getNestDepth() const { return nest.getNestDepth(); }

  /// Get the outermost loop in the nest i.e. the root of the nest. This is
  /// guaranteed to be a tapir loop.
  Loop &getOutermostLoop() const { return nest.getOutermostLoop(); }

  /// Get all the loops in the nest. These will include both tapir and
  /// non-tapir loops.
  ArrayRef<Loop *> getLoops() const { return nest.getLoops(); }

  /// Get the function to which this tapir loop nest belongs.
  Function *getParent() const { return nest.getParent(); }

public:
  /// Create a tapir loop nest object rooted at the given loop. If the loop is
  /// not a tapir loop, this will return nullptr.
  /// Consider the loop shown here
  ///
  /// \code
  ///     forall (i ...)
  ///       forall (j ...)
  ///         forall (k ...)
  /// \endcode
  ///
  /// Here, tapir loop nest objects can be created that are rooted at any of
  /// the forall loops since each of these is a valid tapir loop nest.
  ///
  static std::unique_ptr<TapirLoopNest> create(Loop &loop, ScalarEvolution &se,
                                               TaskInfo &ti);
};

} // namespace llvm

#endif // KITSUNE_ANALYSIS_TAPIR_LOOP_NEST_ANALYSIS_H
