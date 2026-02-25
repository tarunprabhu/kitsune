//===- LoopUtils.h - Utilities for LLVM loops ------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM loops.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "llvm/Analysis/LoopInfo.h"

using namespace llvm;

void llvm::clearTapirLoopAttrs(Loop &loop) {
  LLVMContext &ctx = loop.getHeader()->getContext();
  MDNode *loopMD = loop.getLoopID();
  MDNode *newLoopMD = makePostTransformationMetadata(
      ctx, loopMD, {tapirLoopAttrNamePrefix}, {});

  loop.setLoopID(newLoopMD);
}

static void collectSubLoops(Loop &loop, SmallVector<Loop *, 4> &subLoops) {
  for (Loop *subLoop : loop.getSubLoops()) {
    subLoops.push_back(subLoop);
    collectSubLoops(*subLoop, subLoops);
  }
}

SmallVector<Loop *, 4> llvm::getAllSubLoops(Loop &loop) {
  SmallVector<Loop *, 4> subLoops;
  collectSubLoops(loop, subLoops);

  return subLoops;
}
