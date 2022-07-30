//=- AsyncPrefetch.cpp - Asynchronous prefetching for Tapir loops -*- C++ -*-=/
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Adds asynchronous prefetch calls when appropriate before Tapir loops.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/AsyncPrefetch.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#include <map>
#include <set>
#include <vector>

using namespace llvm;

// forall loops L1, L2, ... Ln are deemed to be siblings if they share the
// same parent loop Lp or if none of L1, L2 ... Ln have a parent loop i.e. if
// all Li are top-level loops within the function. There could be an arbitrary
// amount of code between each Li even if they are siblings.
using SiblingForalls = std::vector<const Loop *>;

class AsyncPrefetchImpl {
private:
  LoopInfo &LI;
  TaskInfo &TI;
  AliasAnalysis &AA;

private:
  bool isAdjacent(const Loop* li, const Loop *lj);
  bool isForall(const Loop *loop);
  std::vector<SiblingForalls> getSiblingForalls();

public:
  AsyncPrefetchImpl(LoopInfo &LI, TaskInfo &TI, AliasAnalysis &AA)
      : LI(LI), TI(TI), AA(AA) {
    ;
  }

  void runOnFunction(Function &F);
};

bool AsyncPrefetchImpl::isForall(const Loop *loop) {
  return getTaskIfTapirLoop(loop, &TI);
}

std::vector<SiblingForalls> AsyncPrefetchImpl::getSiblingForalls() {
  std::map<const Loop *, SiblingForalls> m;
  for (const Loop *loop : LI.getLoopsInPreorder()) {
    if (isForall(loop))
      m[loop->getParentLoop()].push_back(loop);
  }

  std::vector<SiblingForalls> ret;
  for (auto &i : m)
    ret.push_back(i.second);
  return ret;
}

bool AsyncPrefetchImpl::isAdjacent(const Loop* li, const Loop* lj) {
  // An exit block is a block outside the loop to which the loop body
  // branches when the loop terminates. We require the loop to have a single
  // exit block;
  BasicBlock* exit = li->getExitBlock();

  // The preheader is the unique block which branches to the loop header.
  BasicBlock* preheader = lj->getLoopPreheader();

  if (exit and preheader) {
    BasicBlock* succ = exit->getSingleSuccessor();
    BasicBlock* pred = preheader->getSinglePredecessor();
    if (succ and pred) {
      if (succ == pred) {
        llvm::errs() << li->getName() << " => " << lj->getName() << "\n";
        llvm::errs() << "exit\n" << *exit << "\n";
        llvm::errs() << "common\n" << *succ << "\n";
        llvm::errs() << "preheader\n" << *preheader << "\n";
        return true;
      }
    }
  }

  return false;
}

void AsyncPrefetchImpl::runOnFunction(Function &F) {
  llvm::errs() << "loops\n";
  for (const Loop* loop : LI.getLoopsInPreorder())
    llvm::errs() << "  " << loop << " [" << loop->getName() << "]\n";

  for (const SiblingForalls &siblings : getSiblingForalls()) {
    llvm::errs() << "begin sibling group\n";
    for (const Loop* loop : siblings)
      llvm::errs() << "  " << loop->getName() << "\n";
    std::vector<const Loop*> adj;
    for (size_t i = 0; i < siblings.size() - 1; i++) {
      const Loop* li = siblings[i];
      const Loop* lj = siblings[i + 1];
      isAdjacent(li, lj);
    }
    for (const Loop *loop : siblings) {
      BasicBlock *header = loop->getLoopPreheader() ? loop->getLoopPreheader()
                                                    : loop->getHeader();
      BasicBlock *exit = loop->getExitBlock();

      // Two loops Li and Lj are adjacent if they are siblings and there is no
      // executable code except an unconditional branch between the exit block
      // of Li and the preheader of Lj.
      for (BasicBlock *pred : predecessors(header))
        llvm::errs() << "pred: " << pred << " => [" << pred->getName() << "] " << LI[pred] << "\n";
      llvm::errs() << "hder: " << header << " => [" << header->getName()
                   << "]\n";
      llvm::errs() << "exit: " << exit << " => [" << exit->getName() << "]\n";
      for (BasicBlock *succ : successors(exit))
        llvm::errs() << "succ: " << succ << " => [" << succ->getName() << "] " << LI[succ] << "\n";
      llvm::errs() << "---------------\n";
    }
    llvm::errs() << "end sibling group\n\n";
  }
}

PreservedAnalyses AsyncPrefetchPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  llvm::errs() << "AsyncPrefetchPass: " << F.getName() << "\n";

  AliasAnalysis &AA = AM.getResult<AAManager>(F);
  LoopInfo &LI = AM.getResult<LoopAnalysis>(F);
  TaskInfo &TI = AM.getResult<TaskAnalysis>(F);

  AsyncPrefetchImpl(LI, TI, AA).runOnFunction(F);

  // // Not sure if other analyses are also preserved.
  // PreservedAnalyses PA;
  // PA.preserve<LoopAnalysis>();
  // PA.preserve<TaskAnalysis>();

  // return PA;
  return PreservedAnalyses::none();
}
