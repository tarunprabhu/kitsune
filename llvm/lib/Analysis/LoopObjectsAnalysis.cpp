//===- LoopObjectsAnalysis.cpp - Loop Objects Analysis Implementation
//-------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The implementation for the loop objects analysis that was originally
// developed for use by the Kitsune Cuda backend.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopObjectsAnalysis.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Tapir.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

LoopObjectsInfo::LoopObjectsInfo(Loop *L, const TargetLibraryInfo *TLI,
                                 TaskInfo *TI)
    : TheLoop(L), analyzed(canAnalyzeLoop(TI)) {
  if (analyzed)
    analyzeLoop(TLI);
}

const std::vector<Value *> &LoopObjectsInfo::getReadObjects() const {
  return ReadObjects;
}

const std::vector<Value *> &LoopObjectsInfo::getWriteObjects() const {
  return WriteObjects;
}

bool LoopObjectsInfo::hasInfo() const {
  return analyzed;
}

bool LoopObjectsInfo::canAnalyzeLoop(TaskInfo *TI) const {
  return TI and getTaskIfTapirLoop(TheLoop, TI);
}

void LoopObjectsInfo::analyzeLoop(const TargetLibraryInfo *TLI) {
  ReadObjects = getUsedObjects<LoadInst>(TheLoop->getBlocks(), TLI);
  WriteObjects = getUsedObjects<StoreInst>(TheLoop->getBlocks(), TLI);
}

// This function is kept separate because the way to check the return attribute
// will change when upgrading to a newer version of LLVM.
static bool hasReturnAttr(Function &Func, Attribute::AttrKind attr) {
  return Func.getAttributes().hasAttribute(0, attr);
}

bool LoopObjectsInfo::isAllocator(Function *Func,
                                  const TargetLibraryInfo *TLI) {
  if (isMallocOrCallocLikeFn(Func, TLI, true))
    return true;
  else if (hasReturnAttr(*Func, Attribute::NoAlias))
    // FIXME: Not sure if it is necessarily true that just because a function
    // has a noalias attribute on its return value that it is an allocator
    // function. For C++, restrict on the return type has no effect, but not
    // sure if this will be true in other languages.
    return true;
  return false;
}

// This currently traces the pointer operand of a load/store instruction back
// some distance. Currently, it is expected that the operand will be the
// result of a memory allocation instruction/call, a local variable (alloca) or
// a function parameter.
//
// FIXME: Support cases where the pointer is an element of a struct or an
// array. It is not clear exactly what this implies at a higher level. This
// is intended to be used with code that is being compiled for the GPU. The
// situations described here would suggest that some data structure consisting
// of other pointers is resident on the GPU. In principle, this is not
// unreasonable, especially if UVM is in use, but it is a more complicated case
// and it is being deferred for now in the interest of getting something simple
// that will work.
Value *LoopObjectsInfo::trace(Value *V, const TargetLibraryInfo *TLI) {
  if (isa<Argument>(V)) {
    return V;
  } else if (isa<AllocaInst>(V)) {
    return V;
  } else if (auto *Cast = dyn_cast<CastInst>(V)) {
    return trace(Cast->getOperand(0), TLI);
  } else if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
    // For a GEP instruction, if this is an array lookup, it will have a single
    // index operand. Anything else will be a lookup into a more complicated
    // structure that is currently not supported.
    if (GEP->getNumIndices() == 1)
      return trace(GEP->getPointerOperand(), TLI);
  } else if (auto *Call = dyn_cast<CallBase>(V)) {
    // This implies that the pointer was returned by function. For now, the
    // function is expected to be malloc-like in the sense that the return
    // value must be annotated "noalias" which implies that the returned
    // pointer can be assumed to be different from any other pointer alive in
    // the system.
    //
    // FIXME: This will not work for an indirect function call. It is possible
    // that some indirect functions can be resolved in a reasonable way. For
    // instance, if this is a virtual function call, we may be able to determine
    // which vtables this function call will lookup and we may be able to
    // get a set of functions that are called here.
    if (Function *Callee = Call->getCalledFunction()) {
      if (isAllocator(Callee, TLI))
        return Call;
    }
  }
  return nullptr;
}

template <typename I>
std::vector<Value *>
LoopObjectsInfo::getUsedObjects(ArrayRef<BasicBlock *> BBS,
                                const TargetLibraryInfo *TLI) {
  std::vector<Value *> Objects;

  for (BasicBlock *BB : BBS) {
    for (Instruction &Inst : *BB) {
      if (auto *MemInst = dyn_cast<I>(&Inst)) {
        Value *Object = trace(MemInst->getPointerOperand(), TLI);
        llvm::errs() << MemInst->getOpcodeName() << ": " << *MemInst << "\n"
                     << "  " << *MemInst->getPointerOperand() << "\n";
        if (Object)
          llvm::errs() << "  " << *Object << "\n";
        else
          llvm::errs() << "   (nullptr)\n";
        assert(Object && "Could not trace pointer argument.");
        Objects.push_back(Object);
      }
    }
  }

  return Objects;
}

AnalysisKey LoopObjectsAnalysis::Key;

LoopObjectsInfo LoopObjectsAnalysis::run(Loop &L, LoopAnalysisManager &AM,
                                         LoopStandardAnalysisResults &AR) {
  return LoopObjectsInfo(&L, &AR.TLI, &AR.TI);
}
