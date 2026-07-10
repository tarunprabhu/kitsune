//===- QthreadsTT.h - Tapir target that lowers to Qthreads -----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tapir target that lowers to Qthreads.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TARGETS_QTHREADS_TT_H
#define KITSUNE_TARGETS_QTHREADS_TT_H

#include "CPUTTCommon.h"

namespace llvm {

/// \ingroup kitsune
/// Tapir target that lowers to Qthreads via a thin wrapper provided by Kitsune.
/// The underlying qthreads runtime determines how to split the iterations of
/// a parallel loop across available compute elements ("shepherds" in qthreads
/// terminology).
class QthreadsTT : public CPUTTBase {
public:
  QthreadsTT(Module &m, const TTOptions &ttOpts);
  virtual ~QthreadsTT() = default;

  /// Create a custom loop outline processor for this tapir target.
  LoopOutlineProcessor *
  getLoopOutlineProcessor(const TapirLoopInfo *tl) override final;
};

} // namespace llvm

#endif // KITSUNE_TARGETS_QTHREADS_TT_H
