//===- TapirLoopUtils.cpp - Utilities for tapir loops ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for tapir loops.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/TapirLoopUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"

using namespace llvm;

PHINode *llvm::getPrimaryIV(const TapirLoopInfo &tapirLoop) {
  return tapirLoop.getPrimaryInduction().first;
}

Instruction *llvm::getPrimaryIVInc(const TapirLoopInfo &tapirLoop) {
  BasicBlock *latch = tapirLoop.getLoop()->getLoopLatch();
  PHINode *iv = getPrimaryIV(tapirLoop);

  assert(latch && "Expected unique latch in tapir loop");
  assert(iv && "Expected tapir loop to have a primary induction variable");
  if (!latch || !iv)
    return nullptr;

  Value *incoming = iv->getIncomingValueForBlock(latch);
  assert(incoming && isa<Instruction>(incoming) &&
         "Incoming value from tapir loop latch must be an instruction");

  return cast<Instruction>(incoming);
}
