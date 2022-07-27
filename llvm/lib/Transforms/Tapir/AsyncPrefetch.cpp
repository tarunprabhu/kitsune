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

using SiblingForalls = std::vector<const Loop *>;

class AsyncPrefetchImpl {
private:
  LoopInfo &LI;
  TaskInfo &TI;
  AliasAnalysis &AA;

private:
  void getLoops(const Loop *loop, std::set<const Loop *> &loops);
  std::vector<const Loop *> getLoops();
  bool isForall(const Loop *loop);
  std::vector<SiblingForalls> getSiblingForalls();

public:
  AsyncPrefetchImpl(LoopInfo &LI, TaskInfo &TI, AliasAnalysis &AA)
      : LI(LI), TI(TI), AA(AA) {
    ;
  }

  void runOnFunction(Function &F);
};

void AsyncPrefetchImpl::getLoops(const Loop *loop,
                                 std::set<const Loop *> &loops) {
  if (loops.find(loop) != loops.end())
    return;

  loops.insert(loop);
  for (const Loop *subLoop : *loop)
    getLoops(subLoop, loops);
}

std::vector<const Loop *> AsyncPrefetchImpl::getLoops() {
  std::set<const Loop *> loops;
  for (const Loop *loop : LI)
    getLoops(loop, loops);
  return std::vector<const Loop *>(loops.begin(), loops.end());
}

bool AsyncPrefetchImpl::isForall(const Loop *loop) {
  return getTaskIfTapirLoop(loop, &TI);
}

std::vector<SiblingForalls> AsyncPrefetchImpl::getSiblingForalls() {
  std::map<const Loop *, SiblingForalls> m;
  for (const Loop *loop : getLoops()) {
    if (isForall(loop))
      m[loop->getParentLoop()].push_back(loop);
  }

  std::vector<SiblingForalls> ret;
  for (auto &i : m)
    ret.push_back(i.second);
  return ret;
}

void AsyncPrefetchImpl::runOnFunction(Function &F) {
  for (const SiblingForalls &siblings : getSiblingForalls()) {
    llvm::errs() << "begin sibling group\n";
    std::set<Value *> all;
    for (const Loop *loop : siblings) {
      BasicBlock *header = loop->getLoopPreheader() ? loop->getLoopPreheader()
                                                    : loop->getHeader();
      BasicBlock *exit = loop->getExitBlock();

      for (BasicBlock *pred : predecessors(header))
        llvm::errs() << "pred: " << pred << " => [" << pred->getName() << "]\n";
      llvm::errs() << "hder: " << header << " => [" << header->getName() << "]\n";
      llvm::errs() << "exit: " << exit << " => [" << exit->getName() << "]\n";
      for (BasicBlock *succ : successors(exit))
        llvm::errs() << "succ: " << succ << " => [" << succ->getName() << "]\n";
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
