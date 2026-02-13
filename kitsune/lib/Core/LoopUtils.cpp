//===- LoopUtils.cpp - Utilities for LLVM loops ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM Loop's.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/LoopUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"

using namespace llvm;

static void setLoopMD(Loop &loop, StringRef name, unsigned val) {
  LLVMContext &ctx = loop.getHeader()->getContext();

  Type *i32 = Type::getInt32Ty(ctx);
  Constant *c = ConstantInt::get(i32, val);
  Metadata *mdTag = MDString::get(ctx, name);
  Metadata *mdVal = ConstantAsMetadata::get(c);
  MDNode *md = MDNode::get(ctx, {mdTag, mdVal});
  MDNode *loopMD = loop.getLoopID();
  MDNode *newLoopMD = makePostTransformationMetadata(ctx, loopMD, {name}, {md});

  loop.setLoopID(newLoopMD);
}

void llvm::setTapirLoopPerfectDepthMD(Loop &loop, unsigned depth) {
  setLoopMD(loop, loopMDNamePerfectDepth, depth);
}

void llvm::setTapirLoopPerfectLevelMD(Loop &loop, unsigned depth) {
  setLoopMD(loop, loopMDNamePerfectLevel, depth);
}

template <typename T> static T getTapirLoopMD(const Loop &loop, StringRef name);

template <> unsigned getTapirLoopMD(const Loop &loop, StringRef name) {
  MDNode *md = findOptionMDForLoop(&loop, name);
  if (md && md->getNumOperands() == 2)
    if (auto* cmd = dyn_cast<ConstantAsMetadata>(md->getOperand(1)))
      if (auto* c = dyn_cast<ConstantInt>(cmd->getValue()))
        return c->getLimitedValue();
  return 0;
}

unsigned llvm::getTapirLoopPerfectDepthMD(const Loop &loop) {
  return getTapirLoopMD<unsigned>(loop, loopMDNamePerfectDepth);
}

unsigned llvm::getTapirLoopPerfectLevelMD(const Loop &loop) {
  return getTapirLoopMD<unsigned>(loop, loopMDNamePerfectLevel);
}
