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

static constexpr StringRef attrLoopUnrollDisable = "llvm.loop.unroll.disable";

/// Certain LLVM loop attributes are required on all tapir loops. This is that
/// list.
static constexpr StringRef mandatoryLLVMLoopAttrs[] = {
    // Disable unrolling on all tapir loops. The OpenCilk compiler relies on the
    // tapir-to-target pass to handle the multiple detach instructions that
    // result from unrolling a tapir loop. However, Kitsune's tapir targets
    // operate primarily on loop nests and require these to have a single detach
    // instruction. Disabling unrolling is the only way to ensure that,
    // especially at higher optimization levels, we do not end up with a tapir
    // loop nest that Kitsune's tapir targets are unable to process.
    attrLoopUnrollDisable,
};

// Add metadata spelled \p attr to the loop metadata of \p loop.
static void addLoopAttr(Loop &loop, StringRef attr) {
  LLVMContext &ctx = loop.getHeader()->getContext();
  MDNode *loopMD = loop.getLoopID();
  MDNode *attrMD = MDNode::get(ctx, MDString::get(ctx, attr));

  // Remove the existing attribute before adding it back in. I think this
  // ensures that the attribute does not get duplicated.
  MDNode *newLoopMD =
      makePostTransformationMetadata(ctx, loopMD, {attr}, {attrMD});

  loop.setLoopID(newLoopMD);
}

// Remove the metadata spelled \p attr from the loop metadata of \p loop. If
// \p attr does not exist in the loop metadata, this has no effect.
static void clearLoopAttr(Loop &loop, StringRef attr) {
  LLVMContext &ctx = loop.getHeader()->getContext();
  MDNode *loopMD = loop.getLoopID();
  MDNode *newLoopMD = makePostTransformationMetadata(ctx, loopMD, {attr}, {});

  loop.setLoopID(newLoopMD);
}

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

Module *llvm::getModule(Loop &loop) {
  if (Function *f = getFunction(loop))
    return f->getParent();
  return nullptr;
}

const Module *llvm::getModule(const Loop &loop) {
  if (const Function *f = getFunction(loop))
    return f->getParent();
  return nullptr;
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

void llvm::addMandatoryLLVMLoopAttrs(Loop &loop) {
  assert(isTapirLoop(loop));

  for (StringRef attr : mandatoryLLVMLoopAttrs) {
    // If any attributes require special handling, deal with them here. These
    // will typically be those that have related attributes that must also be
    // handled. For instance, when disabling unrolling, we must also remove
    // other unrolling-related attributes such as "llvm.loop.unroll.count", and
    // "llvm.loop.unroll.enable". If LLVM already provides API's that do this,
    // they should be used.
    //
    // If not special handling is required, addLoopAttr() will work just fine.
    if (attr == attrLoopUnrollDisable)
      loop.setLoopAlreadyUnrolled();
    else
      addLoopAttr(loop, attr);
  }
}

void llvm::clearMandatoryLLVMLoopAttrs(Loop &loop) {
  assert(isTapirLoop(loop));

  for (StringRef attr : mandatoryLLVMLoopAttrs)
    // If any attributes require special handling, do so here. Otherwise, using
    // clearLoopAttr() will work just fine.
    clearLoopAttr(loop, attr);
}

bool llvm::serializeTapirLoop(Loop &loop, Task &task, DominatorTree *dt,
                              TaskInfo *ti) {
  assert(isTapirLoop(loop));

  // This must be called early in case `SerializeDetach` removes something that
  // identifies `loop` as being a tapir loop. `SerializeDetach` should not need
  // any non-tapir loop attributes.
  clearMandatoryLLVMLoopAttrs(loop);

  // This performs the actual serialization of the loop.
  SerializeDetach(task.getDetach(), &task, /*ReplaceWithTaskFrame=*/false, dt,
                  ti);

  // Clear the tapir loop attributes *AFTER* calling clearMandatoryLLVMLoopAttrs
  // since that function requires the tapir loop attributes to be present. This
  // is called after Serializedetach in case that function examines the tapir
  // loop attributes.
  clearTapirLoopAttrs(loop);

  return true;
}

unsigned llvm::getNumIndVars(const Loop &loop) {
  iterator_range<BasicBlock::phi_iterator> phis = loop.getHeader()->phis();
  return std::distance(phis.begin(), phis.end());
}
