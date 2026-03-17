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
#include "kitsune/Core/DIUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "llvm/Analysis/LoopInfo.h"

using namespace llvm;

Function *llvm::getFunction(Loop &loop) {
  return loop.getHeader()->getParent();
}

const Function *llvm::getFunction(const Loop &loop) {
  return loop.getHeader()->getParent();
}

std::string llvm::getName(const Loop &loop, StringRef defawlt) {
  if (std::optional<StringRef> name = getNameAttr(loop))
    return name->str();
  else if (DebugLoc dbgLoc = loop.getStartLoc())
    return toString(dbgLoc, /*inlinedAt=*/false);
  else if (loop.getHeader()->hasName())
    return loop.getHeader()->getName().str();
  else
    return defawlt.str();
}

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

SmallVector<BasicBlock *, 8> llvm::getBlocksNotInSubLoops(const Loop &loop) {
  SmallPtrSet<BasicBlock *, 8> bbsInSubLoops;
  for (Loop *subLoop : loop.getSubLoops())
    for (BasicBlock *bb : subLoop->blocks())
      bbsInSubLoops.insert(bb);

  SmallVector<BasicBlock *, 8> bbsInLoop;
  for (BasicBlock *bb : loop.blocks())
    if (!bbsInSubLoops.contains(bb))
      bbsInLoop.push_back(bb);

  return bbsInLoop;
}

BasicBlock* llvm::getUniqueBackEdge(const Loop &loop) {
  BasicBlock *incoming = nullptr, *backedge = nullptr;
  if (loop.getIncomingAndBackEdge(incoming, backedge))
    return backedge;
  return nullptr;
}
