//===- PthreadsTT.h - Tapir target that lowers to pthreads -----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tapir target that lowers to POSIX threads (pthreads).
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TARGETS_PTHREADS_TT_H
#define KITSUNE_TARGETS_PTHREADS_TT_H

#include "CPUTTCommon.h"

namespace llvm {

class TTOptions;

/// \ingroup kitsune
/// Tapir target that splits the iterations of tapir loops across a number of
/// POSIX threads.
class PthreadsTT : public CPUTTBase {
public:
  PthreadsTT(Module &m, const TTOptions &ttOpts);
  virtual ~PthreadsTT() = default;

  /// Create a custom loop outline processor for this tapir target.
  LoopOutlineProcessor *
  getLoopOutlineProcessor(const TapirLoopInfo *tl) override final;
};

} // namespace llvm

#endif // KITSUNE_TARGETS_PTHREADS_TT_H
