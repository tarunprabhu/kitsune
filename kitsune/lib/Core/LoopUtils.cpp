//===- LoopUtils.cpp - Utilities for LLVM loops ---------------------------===//
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
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

LLVMContext &llvm::getContext(const Loop &loop) {
  assert(loop.getHeader() && "Loop does not have a header");
  return loop.getHeader()->getContext();
}

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

template <typename L>
static void collectSubLoops(L &loop, SmallVector<L *, 4> &subLoops) {
  for (L *subLoop : loop.getSubLoops()) {
    subLoops.push_back(subLoop);
    collectSubLoops(*subLoop, subLoops);
  }
}

template <typename L> static SmallVector<L *, 4> getAllSubLoops(L &loop) {
  SmallVector<L *, 4> subLoops;
  collectSubLoops(loop, subLoops);

  return subLoops;
}

SmallVector<Loop *, 4> llvm::getAllSubLoops(Loop &loop) {
  return ::getAllSubLoops<Loop>(loop);
}

SmallVector<const Loop *, 4> llvm::getAllSubLoops(const Loop &loop) {
  return ::getAllSubLoops<const Loop>(loop);
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

BasicBlock *llvm::getUniqueBackEdge(const Loop &loop) {
  BasicBlock *incoming = nullptr, *backedge = nullptr;
  if (loop.getIncomingAndBackEdge(incoming, backedge))
    return backedge;
  return nullptr;
}

bool llvm::isTapirLoop(const Loop &loop) { return hasTargetAttr(loop); }

// Return true if any of the ancestors of a loop are tapir loops. The given
// loop is not required to be a tapir loop. If the given loop is a top-level
// loop, return false.
static bool isAnyAncestorTapirLoop(const Loop &loop) {
  Loop *parentLoop = loop.getParentLoop();
  if (!parentLoop)
    return false;
  else if (isTapirLoop(*parentLoop))
    return true;
  else
    return isAnyAncestorTapirLoop(*parentLoop);
}

bool llvm::isTopLevelTapirLoop(const Loop &loop) {
  return isTapirLoop(loop) && not isAnyAncestorTapirLoop(loop);
}

bool llvm::isTapirLoopForGPU(const Loop &loop) {
  if (!isTapirLoop(loop))
    return false;

  TTID tt = *getTargetAttr(loop);
  if (tt != TTID::Cuda && tt != TTID::Hip)
    return false;

  for (const Loop *subLoop : getAllSubLoops(loop))
    if (isTapirLoop(*subLoop))
      if (getTargetAttr(*subLoop) != tt)
        return false;

  return true;
}

bool llvm::isTopLevelTapirLoopForGPU(const Loop &loop) {
  return isTopLevelTapirLoop(loop) && isTapirLoopForGPU(loop);
}

SmallVector<Loop *, 4> llvm::getTopLevelTapirLoops(LoopInfo &li) {
  SmallVector<Loop *, 4> loops;
  for (Loop *loop : li.getLoopsInPreorder())
    if (isTopLevelTapirLoop(*loop))
      loops.push_back(loop);
  return loops;
}

SmallVector<Loop *, 4> llvm::getTapirLoops(LoopInfo &li) {
  SmallVector<Loop *, 4> loops;
  for (Loop *loop : li.getLoopsInPreorder())
    if (isTapirLoop(*loop))
      loops.push_back(loop);
  return loops;
}
