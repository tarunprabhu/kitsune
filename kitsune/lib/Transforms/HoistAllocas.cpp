//===- HoistAllocas.cpp - Hoist allocas to the function entry block -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hoist allocas to the start of the entry block of the function. Allocas that
// are not in the function entry block will be moved before the first non-alloca
// instruction in the function entry block. If any allocas are present in the
// entry block following this non-alloca instruction, they, too will be moved
// before.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/HoistAllocas.h"
#include "llvm/IR/Instructions.h"

#define DEBUG_TYPE "kit-hoist-allocas"

using namespace llvm;

static bool shouldHoist(const AllocaInst &alloca) {
  // If any operand of the alloca is an instruction, then the alloca cannot be
  // hoisted. In all other cases, it should be safe to hoist.
  for (const Value *op : alloca.operands())
    if (isa<Instruction>(op))
      return false;
  return true;
}

/// Hoist any allocas that are not in the entry block of function \p f to the
/// entry block. Returns true if at least one alloca was moved, false otherwise.
static bool hoistAllocas(Function &f) {
  BasicBlock &entry = f.getEntryBlock();
  BasicBlock::iterator insertPt = entry.getFirstNonPHIOrDbgOrAlloca();

  // First, look for any allocas in the entry block that are after the first
  // non-alloca instruction.
  SmallVector<AllocaInst *> allocas;
  for (BasicBlock::iterator i = insertPt, e = entry.end(); i != e; ++i)
    if (auto *alloca = dyn_cast<AllocaInst>(&*i))
      if (shouldHoist(*alloca))
        allocas.push_back(alloca);

  // Then, look for allocas in the rest of the function.
  Function::iterator entryIt = entry.getIterator();
  for (Function::iterator bb = ++entryIt, e = f.end(); bb != e; ++bb)
    for (Instruction &inst : *bb)
      if (auto *alloca = dyn_cast<AllocaInst>(&inst))
        if (shouldHoist(*alloca))
          allocas.push_back(alloca);

  for (AllocaInst *alloca : allocas)
    alloca->moveBefore(insertPt);

  return allocas.size();
}

PreservedAnalyses HoistAllocasPass::run(Function &f,
                                        FunctionAnalysisManager &am) {
  hoistAllocas(f);

  // This is a one-to-one replacement of allocas. It shouldn't have any effect
  // on any analyses, including alias analyses.
  return PreservedAnalyses::all();
}

bool EmbHoistAllocasPass::run(TTID tt, Module &devM, Module &hostM,
                              ModuleAnalysisManager &hostMAM) {
  bool changed = false;
  for (Function &devF : devM.functions())
    if (devF.size())
      changed |= hoistAllocas(devF);
  return changed;
}
