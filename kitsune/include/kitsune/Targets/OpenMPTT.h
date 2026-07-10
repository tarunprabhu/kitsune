//=- OpenMPTT.h - Tapir target that lowers to the OpenMP runtime --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tapir target that lowers to LLVM's OpenMP runtime.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TARGETS_OPENMP_TT_H
#define KITSUNE_TARGETS_OPENMP_TT_H

#include "CPUTTCommon.h"

namespace llvm {

/// Tapir target that lowers tapir constructs to use LLVM's OpenMP runtime.
/// \ingroup kitsune
class OpenMPTT : public CPUTTBase {
public:
  OpenMPTT(Module &m, const TTOptions &ttOpts);
  virtual ~OpenMPTT() = default;

  /// Create a custom loop outline processor for this tapir target.
  LoopOutlineProcessor *
  getLoopOutlineProcessor(const TapirLoopInfo *tl) override final;
};

} // namespace llvm

#endif // KITSUNE_TARGETS_OPENMP_TT_H
