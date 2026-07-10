//===- OpenMPTT.cpp - Tapir target that lowers to LLVM's OpenMP runtime ---===//
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

#include "kitsune/Targets/OpenMPTT.h"
#include "CPUTTLoop.h"

#define DEBUG_TYPE "openmptt"

using namespace llvm;

namespace {

/// \ingroup kitsune
class OpenMPLoopProcessor : public CPUTTLoopProcessor {
public:
  OpenMPLoopProcessor(Module &m, const TTOptions &tto)
      : CPUTTLoopProcessor(TTID::OpenMP, tto, /*asyncLaunch=*/false, m) {}
  virtual ~OpenMPLoopProcessor() = default;
};

} // namespace

OpenMPTT::OpenMPTT(Module &m, const TTOptions &tto) : CPUTTBase(m, tto) {}

LoopOutlineProcessor *
OpenMPTT::getLoopOutlineProcessor(const TapirLoopInfo *tl) {
  return new OpenMPLoopProcessor(M, this->getOptions());
}
