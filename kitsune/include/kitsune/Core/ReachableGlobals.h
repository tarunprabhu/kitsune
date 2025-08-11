//===- ReachableGlobals.h - Collect reachable GlobalValues -----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Analysis to determine the GlobalValue's reachable from various starting
// points.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_REACHABLE_GLOBALS_H
#define KITSUNE_CORE_REACHABLE_GLOBALS_H

#include "llvm/ADT/SmallSet.h"

namespace llvm {

class BasicBlock;
class BlockAddress;
class Constant;
class Function;
class GlobalAlias;
class GlobalIFunc;
class GlobalValue;
class GlobalVariable;
class Loop;

/// Class to collect the set of GlobalValue's reachable from certain starting
/// entities. Currently, this is only capable of detecting those globals used
/// directly within various elements such as basic blocks and global variable
/// initializers. It cannot detect those GlobalValues that are used indirectly,
/// for instance, by pointer.
///
/// In the future, it may be useful to make this analysis more powerful and at
/// least attempt to detect indirect use of GlobalValue's.
class ReachableGlobals {
public:
  using SeenGlobals = SmallSet<GlobalValue *, 16>;
  using Iterator = SeenGlobals::const_iterator;

private:
  using SeenBasicBlocks = SmallSet<BasicBlock*, 16>;

private:
  /// The global values that have been seen.
  SeenGlobals gvs;

  /// The basic blocks that have been seen.
  SeenBasicBlocks bbs;

private:
  void analyze(BlockAddress &addr);
  void analyze(Constant &c);
  void analyze(GlobalAlias &ga);
  void analyze(GlobalIFunc &f);
  void analyze(GlobalVariable &gv);

public:
  Iterator begin() const { return gvs.begin(); }
  Iterator end() const { return gvs.end(); }

  /// Is the set of reachable globals empty?
  bool empty() const { return gvs.empty(); }

  /// Get the size of the reachable globals set.
  size_t size() const { return gvs.size(); }

  /// Check if the given GlobalValue is known to be reachable.
  bool contains(const GlobalValue &g) const { return gvs.contains(&g); }

  /// Collect the GlobalValues used in a \ref BasicBlock.
  void analyze(BasicBlock &bb);

  /// Collect the GlobalValues used in a \ref Function.
  void analyze(Function &f);

  /// Collect the GlobalValues used in a \ref Loop.
  void analyze(Loop &loop);
};

} // namespace llvm

#endif // KITSUNE_CORE_REACHABLE_GLOBALS_H
