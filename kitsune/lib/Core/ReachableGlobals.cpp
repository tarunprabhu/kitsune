//===- ReachableGlobals.cpp - Collect reachable GlobalValues --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Analysis to determine the GlobalValue's reachable from various starting
// points.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ReachableGlobals.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalIFunc.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"

using namespace llvm;

void ReachableGlobals::analyze(GlobalVariable &g) {
  seen.insert(&g);
  if (g.hasInitializer())
    analyze(*g.getInitializer());
}

void ReachableGlobals::analyze(GlobalIFunc &g) {
  seen.insert(&g);
  llvm_unreachable("ReachableGlobals: GNU IFUNC not yet supported");
}

void ReachableGlobals::analyze(GlobalAlias &g) {
  seen.insert(&g);
  llvm_unreachable("ReachableGlobals: GlobalAlias not yet supported");
}

void ReachableGlobals::analyze(BlockAddress &blkaddr) {
  if (Function *f = blkaddr.getFunction())
    analyze(*f);
  if (BasicBlock *bb = blkaddr.getBasicBlock())
    analyze(*bb);
}

void ReachableGlobals::analyze(Constant &c) {
  if (GlobalValue *g = dyn_cast<GlobalValue>(&c))
    if (seen.find(g) != seen.end())
      return;

  if (auto *f = dyn_cast<Function>(&c))
    return analyze(*f);
  else if (auto *g = dyn_cast<GlobalVariable>(&c))
    return analyze(*g);
  else if (auto *g = dyn_cast<GlobalAlias>(&c))
    return analyze(*g);
  else if (auto *g = dyn_cast<GlobalIFunc>(&c))
    return analyze(*g);
  else if (auto *blkaddr = dyn_cast<BlockAddress>(&c))
    return analyze(*blkaddr);
  else
    for (Use &op : c.operands())
      if (auto *cop = dyn_cast<Constant>(op))
        analyze(*cop);
}

void ReachableGlobals::analyze(BasicBlock &bb) {
  for (Instruction &inst : bb)
    for (Use &op : inst.operands())
      if (auto *c = dyn_cast<Constant>(&op))
        analyze(*c);
}

void ReachableGlobals::analyze(Function &f) {
  seen.insert(&f);
  for (BasicBlock &bb : f)
    analyze(bb);
}

void ReachableGlobals::analyze(Loop &loop) {
  // Collect the globals used in any subloops, then the globals used within the
  // loop itself.
  for (Loop *subLoop : loop)
    for (BasicBlock *bb : subLoop->blocks())
      analyze(*bb);
  for (BasicBlock *bb : loop.blocks())
    analyze(*bb);
}
