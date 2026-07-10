//===- PthreadsTT.cpp - Tapir target that lowers to pthreads --------------===//
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

#include "kitsune/Targets/PthreadsTT.h"
#include "CPUTTLoop.h"

#define DEBUG_TYPE "pthreadstt"

using namespace llvm;

namespace {

/// \ingroup kitsune
class PthreadsLoopProcessor : public CPUTTLoopProcessor {
public:
  PthreadsLoopProcessor(Module &m, const TTOptions &tto)
      : CPUTTLoopProcessor(TTID::Pthreads, tto, /*asyncLaunch=*/true, m) {}
  virtual ~PthreadsLoopProcessor() = default;
};

} // namespace

PthreadsTT::PthreadsTT(Module &m, const TTOptions &tto) : CPUTTBase(m, tto) {}

LoopOutlineProcessor *
PthreadsTT::getLoopOutlineProcessor(const TapirLoopInfo *tl) {
  return new PthreadsLoopProcessor(M, this->getOptions());
}
