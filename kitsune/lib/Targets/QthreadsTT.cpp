//===- QthreadsTT.cpp - Tapir target that lowers to Qthreads --------------===//
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

#include "kitsune/Targets/QthreadsTT.h"
#include "CPUTTLoop.h"

#define DEBUG_TYPE "qthreadstt"

using namespace llvm;

namespace {

/// \ingroup kitsune
class QthreadsLoopProcessor : public CPUTTLoopProcessor {
public:
  QthreadsLoopProcessor(Module &m, const TTOptions &tto)
      : CPUTTLoopProcessor(TTID::Qthreads, tto, /*asyncLaunch=*/false, m) {}
  virtual ~QthreadsLoopProcessor() = default;
};

} // namespace

QthreadsTT::QthreadsTT(Module &m, const TTOptions &tto) : CPUTTBase(m, tto) {}

LoopOutlineProcessor *
QthreadsTT::getLoopOutlineProcessor(const TapirLoopInfo *tl) {
  return new QthreadsLoopProcessor(M, this->getOptions());
}
