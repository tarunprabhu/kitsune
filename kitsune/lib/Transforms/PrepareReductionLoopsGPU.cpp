//===- PrepareReductionLoopsGPU.cpp - Transform reduction loops for GPU ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass that transforms tapir-loops that perform reductions to a form that is
// suitable for parallel execution on GPU's.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/PrepareReductionLoopsGPU.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Support/TTIDUtils.h"
#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/WorkSpanAnalysis.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SimplifyIndVar.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"

using namespace llvm;

#define DEBUG_TYPE "kitsune-stripmine-gpu-reductions"

/// Default coarsening factor for strpimined Tapir reduction loops.
static const unsigned defaultCoarseningFactor = 2048;

/// Perform some cleanup and simplifications on loops after stripmining. It is
/// useful to simplify the IV's in the new loop, as well as do a quick
/// simplify/dce pass of the instructions.
static void simplifyLoopAfterStripMine(Loop *L, bool SimplifyIVs, LoopInfo *LI,
                                       ScalarEvolution *SE, DominatorTree *DT,
                                       const TargetTransformInfo &TTI,
                                       AssumptionCache *AC) {
  // Simplify any new induction variables in the stripmined loop.
  if (SE && SimplifyIVs) {
    SmallVector<WeakTrackingVH, 16> DeadInsts;
    simplifyLoopIVs(L, SE, DT, LI, &TTI, DeadInsts);

    // Aggressively clean up dead instructions that simplifyLoopIVs already
    // identified. Any remaining should be cleaned up below.
    while (!DeadInsts.empty())
      if (auto *Inst =
              dyn_cast_or_null<Instruction>(&*DeadInsts.pop_back_val()))
        RecursivelyDeleteTriviallyDeadInstructions(Inst);
  }

  // At this point, the code is well formed.  We now do a quick sweep over the
  // inserted code, doing constant propagation and dead code elimination as we
  // go.
  const DataLayout &DL = L->getHeader()->getModule()->getDataLayout();
  for (BasicBlock *BB : L->getBlocks()) {
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;) {
      Instruction *Inst = &*I++;

      if (Value *V = simplifyInstruction(Inst, {DL, nullptr, DT, AC}))
        if (LI->replacementPreservesLCSSAForm(Inst, V))
          Inst->replaceAllUsesWith(V);
      if (isInstructionTriviallyDead(Inst))
        Inst->eraseFromParent();
    }
  }

  // TODO: after stripmining, previously loop variant conditions are likely to
  // fold to constants, eagerly propagating those here will require fewer
  // cleanup passes to be run.  Alternatively, a LoopEarlyCSE might be
  // appropriate.
}

static Task *getTapirLoopForStripMining(const Loop *L, TaskInfo &TI,
                                        OptimizationRemarkEmitter *ORE) {
  LLVM_DEBUG(dbgs() << "Analyzing for stripmining: " << *L);
  // We only handle Tapir loops.
  Task *T = getTaskIfTapirLoopStructure(L, &TI);
  if (!T)
    return nullptr;

  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader) {
    LLVM_DEBUG(
        dbgs() << "  Can't stripmine: loop preheader-insertion failed.\n");
    if (ORE)
      ORE->emit(
          TapirLoopInfo::createMissedAnalysis(DEBUG_TYPE, "NoPreheader", L)
          << "loop lacks a preheader");
    return nullptr;
  }
  assert(isa<BranchInst>(Preheader->getTerminator()) &&
         "Preheader not terminated by a branch");

  BasicBlock *LatchBlock = L->getLoopLatch();
  if (!LatchBlock) {
    LLVM_DEBUG(
        dbgs() << "  Can't stripmine: loop exit-block-insertion failed.\n");
    if (ORE)
      ORE->emit(TapirLoopInfo::createMissedAnalysis(DEBUG_TYPE, "NoLatch", L)
                << "loop lacks a latch");
    return nullptr;
  }

  // Loops with indirectbr cannot be cloned.
  if (!L->isSafeToClone()) {
    LLVM_DEBUG(dbgs() << "  Can't stripmine: loop body cannot be cloned.\n");
    if (ORE)
      ORE->emit(
          TapirLoopInfo::createMissedAnalysis(DEBUG_TYPE, "UnsafeToClone", L)
          << "loop is not safe to clone");
    return nullptr;
  }

  // Tapir loops where the loop body does not reattach cannot be stripmined.
  if (!llvm::any_of(predecessors(LatchBlock), [](const BasicBlock *B) {
        return isa<ReattachInst>(B->getTerminator());
      })) {
    LLVM_DEBUG(dbgs() << "  Can't stripmine: loop body does not reattach.\n");
    if (ORE)
      ORE->emit(TapirLoopInfo::createMissedAnalysis(DEBUG_TYPE, "NoReattach", L)
                << "spawned loop body does not reattach");
    return nullptr;
  }

  // The current loop-stripmine pass can only stripmine loops with a single
  // latch that's a conditional branch exiting the loop.
  // FIXME: The implementation can be extended to work with more complicated
  // cases, e.g. loops with multiple latches.
  BranchInst *BI = dyn_cast<BranchInst>(LatchBlock->getTerminator());

  if (!BI || BI->isUnconditional()) {
    // The loop-rotate pass can be helpful to avoid this in many cases.
    LLVM_DEBUG(
        dbgs()
        << "  Can't stripmine: loop not terminated by a conditional branch.\n");
    if (ORE)
      ORE->emit(
          TapirLoopInfo::createMissedAnalysis(DEBUG_TYPE, "NoLatchBranch", L)
          << "loop latch is not terminated by a conditional branch");
    return nullptr;
  }

  BasicBlock *Header = L->getHeader();
  auto CheckSuccessors = [&](unsigned S1, unsigned S2) {
    return BI->getSuccessor(S1) == Header && !L->contains(BI->getSuccessor(S2));
  };

  if (!CheckSuccessors(0, 1) && !CheckSuccessors(1, 0)) {
    LLVM_DEBUG(dbgs() << "  Can't stripmine: only loops with one conditional"
                         " latch exiting the loop can be stripmined.\n");
    if (ORE)
      ORE->emit(TapirLoopInfo::createMissedAnalysis(DEBUG_TYPE,
                                                    "ComplexLatchBranch", L)
                << "loop has multiple exiting conditional latches");
    return nullptr;
  }

  if (Header->hasAddressTaken()) {
    // The loop-rotate pass can be helpful to avoid this in many cases.
    LLVM_DEBUG(dbgs() << "  Won't stripmine loop: address of header block is "
                         "taken.\n");
    if (ORE)
      ORE->emit(TapirLoopInfo::createMissedAnalysis(DEBUG_TYPE,
                                                    "HeaderAddressTaken", L)
                << "loop header block has address taken");
    return nullptr;
  }

  // Don't stripmine loops with the convergent attribute.
  for (auto &BB : L->blocks())
    for (auto &I : *BB)
      if (CallBase *CB = dyn_cast<CallBase>(&I))
        if (CB->isConvergent()) {
          LLVM_DEBUG(dbgs() << "  Won't stripmine loop: contains convergent "
                               "attribute.\n");
          if (ORE)
            ORE->emit(TapirLoopInfo::createMissedAnalysis(DEBUG_TYPE,
                                                          "ConvergentLoop", L)
                      << "loop contains convergent attribute");
          return nullptr;
        }

  // Get the task for this loop.
  return T;
}

/// Create a clone of the blocks in a loop and connect them together.
/// If CreateRemainderLoop is false, loop structure will not be cloned,
/// otherwise a new loop will be created including all cloned blocks, and the
/// iterator of it switches to count NewIter down to 0.
/// The cloned blocks should be inserted between InsertTop and InsertBot.
/// If loop structure is cloned InsertTop should be new preheader, InsertBot
/// new loop exit.
/// Return the new cloned loop that is created when CreateRemainderLoop is true.
static Loop *cloneLoopBlocks(
    Loop *L, Value *NewIter, const bool CreateRemainderLoop,
    const bool UseEpilogRemainder, const bool UnrollRemainder,
    BasicBlock *InsertTop, BasicBlock *InsertBot, BasicBlock *Preheader,
    std::vector<BasicBlock *> &NewBlocks, LoopBlocksDFS &LoopBlocks,
    SmallVectorImpl<BasicBlock *> &ExtraTaskBlocks,
    SmallVectorImpl<BasicBlock *> &SharedEHTaskBlocks, ValueToValueMapTy &VMap,
    DominatorTree *DT, LoopInfo *LI, unsigned Count) {
  StringRef Suffix = UseEpilogRemainder ? "epil" : "prol";
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  Function *F = Header->getParent();
  LoopBlocksDFS::RPOIterator BlockBegin = LoopBlocks.beginRPO();
  LoopBlocksDFS::RPOIterator BlockEnd = LoopBlocks.endRPO();
  Loop *ParentLoop = L->getParentLoop();
  NewLoopsMap NewLoops;
  NewLoops[ParentLoop] = ParentLoop;
  if (!CreateRemainderLoop)
    NewLoops[L] = ParentLoop;

  // For each block in the original loop, create a new copy,
  // and update the value map with the newly created values.
  for (LoopBlocksDFS::RPOIterator BB = BlockBegin; BB != BlockEnd; ++BB) {
    BasicBlock *NewBB = CloneBasicBlock(*BB, VMap, "." + Suffix, F);
    NewBlocks.push_back(NewBB);

    // Add the cloned block to loop info.
    addClonedBlockToLoopInfo(*BB, NewBB, LI, NewLoops);

    VMap[*BB] = NewBB;
    if (Header == *BB) {
      // For the first block, add a CFG connection to this newly
      // created block.
      InsertTop->getTerminator()->setSuccessor(0, NewBB);
    }

    if (DT) {
      if (Header == *BB) {
        // The header is dominated by the preheader.
        DT->addNewBlock(NewBB, InsertTop);
      } else {
        // Copy information from original loop to the clone.
        BasicBlock *IDomBB = DT->getNode(*BB)->getIDom()->getBlock();
        DT->addNewBlock(NewBB, cast<BasicBlock>(VMap[IDomBB]));
      }
    }

    if (Latch == *BB) {
      // For the last block, if CreateRemainderLoop is false, create a direct
      // jump to InsertBot. If not, create a loop back to cloned head.
      VMap.erase((*BB)->getTerminator());
      BasicBlock *FirstLoopBB = cast<BasicBlock>(VMap[Header]);
      BranchInst *LatchBR = cast<BranchInst>(NewBB->getTerminator());
      IRBuilder<> Builder(LatchBR);
      if (!CreateRemainderLoop) {
        Builder.CreateBr(InsertBot);
      } else {
        PHINode *NewIdx =
            PHINode::Create(NewIter->getType(), 2, Suffix + ".iter");
        NewIdx->insertBefore(FirstLoopBB->getFirstNonPHIIt());
        Value *IdxSub =
            Builder.CreateSub(NewIdx, ConstantInt::get(NewIdx->getType(), 1),
                              NewIdx->getName() + ".sub");
        Value *IdxCmp =
            Builder.CreateIsNotNull(IdxSub, NewIdx->getName() + ".cmp");
        MDNode *BranchWeights = nullptr;
        if (hasBranchWeightMD(*LatchBR)) {
          uint32_t ExitWeight;
          uint32_t BackEdgeWeight;
          if (Count >= 3) {
            // Note: We do not enter this loop for zero-remainders. The check
            // is at the end of the loop. We assume equal distribution between
            // possible remainders in [1, Count).
            ExitWeight = 1;
            BackEdgeWeight = (Count - 2) / 2;
          } else {
            // Unnecessary backedge, should never be taken. The conditional
            // jump should be optimized away later.
            ExitWeight = 1;
            BackEdgeWeight = 0;
          }
          MDBuilder MDB(Builder.getContext());
          BranchWeights = MDB.createBranchWeights(BackEdgeWeight, ExitWeight);
        }
        Builder.CreateCondBr(IdxCmp, FirstLoopBB, InsertBot, BranchWeights);
        NewIdx->addIncoming(NewIter, InsertTop);
        NewIdx->addIncoming(IdxSub, NewBB);
      }
      LatchBR->eraseFromParent();
    }
  }

  DetachInst *DI = cast<DetachInst>(Header->getTerminator());
  // Create new copies of the EH blocks to clone.  We can handle these blocks
  // more simply than the loop blocks.
  for (BasicBlock *BB : ExtraTaskBlocks) {
    BasicBlock *NewBB = CloneBasicBlock(BB, VMap, "." + Suffix, F);
    NewBlocks.push_back(NewBB);

    // Add the cloned block to loop info.
    if (LI->getLoopFor(BB))
      addClonedBlockToLoopInfo(BB, NewBB, LI, NewLoops);

    VMap[BB] = NewBB;

    // Update PHI nodes in the detach-unwind destination.  Strictly speaking,
    // this step isn't necessary, since the epilog loop will be serialized later
    // and these new entries for the PHI nodes will therefore be removed.  But
    // the routine for serializing the detach expects valid LLVM, so we update
    // the PHI nodes here to ensure the resulting LLVM is valid.
    if (DI->hasUnwindDest()) {
      if (isDetachedRethrow(BB->getTerminator(), DI->getSyncRegion())) {
        InvokeInst *DR = dyn_cast<InvokeInst>(BB->getTerminator());
        for (PHINode &PN : DR->getUnwindDest()->phis())
          PN.addIncoming(PN.getIncomingValueForBlock(BB), NewBB);
      }
    }
  }

  // Update PHI nodes in successors of ExtraTaskBlocks, based on the cloned
  // values.
  for (BasicBlock *BB : ExtraTaskBlocks) {
    for (BasicBlock *Succ : successors(BB)) {
      if (VMap.count(Succ))
        continue;

      for (PHINode &PN : Succ->phis()) {
        Value *Val = PN.getIncomingValueForBlock(BB);
        Value *NewVal = VMap.count(Val) ? cast<Value>(VMap[Val]) : Val;
        PN.addIncoming(NewVal, cast<BasicBlock>(VMap[BB]));
      }
    }
  }

  // Update DT to accommodate cloned ExtraTaskBlocks.
  if (DT) {
    for (BasicBlock *BB : ExtraTaskBlocks) {
      BasicBlock *NewBB = cast<BasicBlock>(VMap[BB]);
      // Copy information from original loop to the clone, if it's available.
      BasicBlock *IDomBB = DT->getNode(BB)->getIDom()->getBlock();
      if (VMap.count(IDomBB)) {
        DT->addNewBlock(NewBB, cast<BasicBlock>(VMap[IDomBB]));
      } else {
        BasicBlock *NewIDom = nullptr;
        // Get the idom of BB's predecessors.
        for (BasicBlock *Pred : predecessors(BB))
          if (VMap.count(Pred)) {
            if (NewIDom)
              NewIDom = DT->findNearestCommonDominator(NewIDom, Pred);
            else
              NewIDom = Pred;
          }
        // Use this computed idom (or its clone) as the idom of the cloned BB.
        if (VMap.count(NewIDom))
          DT->addNewBlock(NewBB, cast<BasicBlock>(VMap[NewIDom]));
        else
          DT->addNewBlock(NewBB, NewIDom);
      }
    }
  }

  // Change the incoming values to the ones defined in the preheader or
  // cloned loop.
  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    PHINode *NewPHI = cast<PHINode>(VMap[&*I]);
    if (!CreateRemainderLoop) {
      if (UseEpilogRemainder) {
        unsigned Idx = NewPHI->getBasicBlockIndex(Preheader);
        NewPHI->setIncomingBlock(Idx, InsertTop);
        NewPHI->removeIncomingValue(Latch, false);
      } else {
        VMap[&*I] = NewPHI->getIncomingValueForBlock(Preheader);
        NewPHI->eraseFromParent();
      }
    } else {
      unsigned Idx = NewPHI->getBasicBlockIndex(Preheader);
      NewPHI->setIncomingBlock(Idx, InsertTop);
      BasicBlock *NewLatch = cast<BasicBlock>(VMap[Latch]);
      Idx = NewPHI->getBasicBlockIndex(Latch);
      Value *InVal = NewPHI->getIncomingValue(Idx);
      NewPHI->setIncomingBlock(Idx, NewLatch);
      if (Value *V = VMap.lookup(InVal))
        NewPHI->setIncomingValue(Idx, V);
    }
  }

  // Add entries to PHI nodes outside of loop.  Strictly speaking, this step
  // isn't necessary, since the epilog loop will be serialized later and these
  // new entries for the PHI nodes will therefore be removed.  But the routine
  // for serializing the detach expects valid LLVM, so we update the PHI nodes
  // here to ensure the resulting LLVM is valid.
  BasicBlock *ClonedHeader = cast<BasicBlock>(VMap[Header]);
  DetachInst *ClonedDetach = cast<DetachInst>(ClonedHeader->getTerminator());
  if (BasicBlock *Unwind = ClonedDetach->getUnwindDest())
    for (PHINode &PN : Unwind->phis())
      PN.addIncoming(PN.getIncomingValueForBlock(Header), ClonedHeader);

  if (CreateRemainderLoop) {
    Loop *NewLoop = NewLoops[L];
    assert(NewLoop && "L should have been cloned");

    // Only add loop metadata if the loop is not going to be completely
    // unrolled.
    if (UnrollRemainder)
      return NewLoop;

    // FIXME?
    // // Add unroll disable metadata to disable future unrolling for this loop.
    // NewLoop->setLoopAlreadyUnrolled();
    return NewLoop;
  }
  return nullptr;
}

// Helper function to get the basic-block predecessors of the given exceptional
// continuation BB associated with task T.  These predecessors are either
// enclosed by task T or come from the unwind of the detach that spawns T.
//
// TODO: Move some of this logic into TapirTaskInfo, so we don't have to
// recompute it?
static void getEHContPredecessors(BasicBlock *BB, Task *T,
                                  SmallVectorImpl<BasicBlock *> &Preds,
                                  TaskInfo &TI) {
  DetachInst *DI = T->getDetach();
  assert(DI && "Root task does not have an exceptional continuation.");
  assert(DI->hasUnwindDest() &&
         "Task does not have an exceptional continuation.");

  // Get the predecessors of BB enclosed by task T.
  for (BasicBlock *Pred : predecessors(BB))
    if (T->encloses(Pred))
      Preds.push_back(Pred);

  // If the unwind destination of the detach is the exceptional continuation BB,
  // add the block that performs the detach and return.
  if (DI->getUnwindDest() == BB) {
    Preds.push_back(DI->getParent());
    return;
  }

  // Get the predecessor that comes from the unwind of the detach.
  BasicBlock *DetUnwind = DI->getUnwindDest();
  while (DetUnwind->getUniqueSuccessor() != BB)
    DetUnwind = DetUnwind->getUniqueSuccessor();
  Preds.push_back(DetUnwind);
}

static bool isReducer(const Function &f) {
  // TODO: At some point, we will implement a reduction intrinsic, and this
  // should be replaced with that. For now, we are testing, so we assume that
  // any function calls are reduction functions.

  // TODO: Obviously, this should be replaced with an actual check.
  return true;
}

static Loop *StripMineReductionLoop(
    Loop *L, unsigned Count, bool AllowExpensiveTripCount, bool UnrollRemainder,
    LoopInfo *LI, ScalarEvolution *SE, DominatorTree *DT,
    const TargetTransformInfo &TTI, AssumptionCache *AC, TaskInfo *TI,
    OptimizationRemarkEmitter *ORE, bool PreserveLCSSA, bool ParallelEpilog,
    bool NeedNestedSync, Loop **RemainderLoop, bool GPU) {
  Task *T = getTapirLoopForStripMining(L, *TI, ORE);
  if (!T)
    return nullptr;

  TapirLoopInfo TL(L, T);

  // Use Scalar Evolution to compute the trip count. This allows more loops to
  // be stripmined than relying on induction var simplification.
  if (!SE)
    return nullptr;
  PredicatedScalarEvolution PSE(*SE, *L);

  TL.collectIVs(PSE, DEBUG_TYPE, ORE);

  // If no primary induction was found, just bail.
  if (!TL.hasPrimaryInduction()) {
    LLVM_DEBUG(dbgs() << "No primary induction variable found in loop.");
    return nullptr;
  }
  PHINode *PrimaryInduction = TL.getPrimaryInduction().first;
  LLVM_DEBUG(dbgs() << "\tPrimary induction " << *PrimaryInduction << "\n");

  Value *TripCount = TL.getOrCreateTripCount(PSE, DEBUG_TYPE, ORE);
  if (!TripCount) {
    LLVM_DEBUG(dbgs() << "Could not compute trip count.\n");
    if (ORE)
      ORE->emit(
          TapirLoopInfo::createMissedAnalysis(DEBUG_TYPE, "NoTripCount", L)
          << "could not compute finite loop trip count.");
    return nullptr;
  }

  LLVM_DEBUG(dbgs() << "\tTrip count " << *TripCount << "\n");

  // Fixup all external uses of the IVs.
  for (auto &InductionEntry : *TL.getInductionVars())
    TL.fixupIVUsers(InductionEntry.first, InductionEntry.second, PSE);

  // High-level algorithm: Generate an epilog for the Tapir loop and insert it
  // between the original latch and its exit.  Then split the entry and reattach
  // block of the loop body to build the serial inner loop.

  BasicBlock *Preheader = L->getLoopPreheader();
  BranchInst *PreheaderBR = cast<BranchInst>(Preheader->getTerminator());
  BasicBlock *Latch = L->getLoopLatch();
  BasicBlock *Header = L->getHeader();
  BasicBlock *TaskEntry = T->getEntry();

  assert(isa<DetachInst>(Header->getTerminator()) &&
         "Header not terminated by a detach.");
  DetachInst *DI = cast<DetachInst>(Header->getTerminator());
  assert(DI->getDetached() == TaskEntry &&
         "Task entry does not match block detached from header.");
  BasicBlock *ParentEntry = T->getParentTask()->getEntry();
  BranchInst *LatchBR = cast<BranchInst>(Latch->getTerminator());
  unsigned ExitIndex = LatchBR->getSuccessor(0) == Header ? 1 : 0;
  BasicBlock *LatchExit = LatchBR->getSuccessor(ExitIndex);

  Function *F = Header->getParent();
  LLVM_DEBUG(dbgs() << "Function before strip mining\n" << *F);

  // We will use the increment of the primary induction variable to derive
  // wrapping flags.
  Instruction *PrimaryInc =
      cast<Instruction>(PrimaryInduction->getIncomingValueForBlock(Latch));

  // Get all uses of the primary induction variable in the task.
  SmallVector<Use *, 4> PrimaryInductionUsesInTask;
  for (Use &U : PrimaryInduction->uses())
    if (Instruction *User = dyn_cast<Instruction>(U.getUser()))
      if (T->encloses(User->getParent()))
        PrimaryInductionUsesInTask.push_back(&U);

  // KITSUNE FIXME: If we know that the loop is performing a reduction, we
  // should not need to care about the trip count. A loop-variant trip count
  // is a semantic error.
  //
  // Only stripmine loops with a computable trip count, and the trip count needs
  // to be an int value (allowing a pointer type is a TODO item).
  // We calculate the backedge count by using getExitCount on the Latch block,
  // which is proven to be the only exiting block in this loop. This is same as
  // calculating getBackedgeTakenCount on the loop (which computes SCEV for all
  // exiting blocks).
  const SCEV *BECountSC = TL.getBackedgeTakenCount(PSE);
  if (isa<SCEVCouldNotCompute>(BECountSC) ||
      !BECountSC->getType()->isIntegerTy()) {
    LLVM_DEBUG(dbgs() << "Could not compute exit block SCEV\n");
    return nullptr;
  }

  unsigned BEWidth =
      cast<IntegerType>(TL.getWidestInductionType())->getBitWidth();

  // Add 1 since the backedge count doesn't include the first loop iteration.
  const SCEV *TripCountSC = TL.getExitCount(BECountSC, PSE);
  if (isa<SCEVCouldNotCompute>(TripCountSC)) {
    LLVM_DEBUG(dbgs() << "Could not compute trip count SCEV.\n");
    return nullptr;
  }

  // This constraint lets us deal with an overflowing trip count easily; see the
  // comment on ModVal below.
  if (Log2_32(Count) > BEWidth) {
    LLVM_DEBUG(
        dbgs()
        << "Count failed constraint on overflow trip count calculation.\n");
    return nullptr;
  }

  // Loop structure is the following:
  //
  // Preheader
  //   Header
  //   ...
  //   Latch
  // LatchExit
  Module *M = F->getParent();

  // Insert the epilog remainder.
  BasicBlock *NewPreheader;
  BasicBlock *NewExit = nullptr;
  BasicBlock *EpilogPreheader = nullptr;
  {
    // Split Preheader to insert a branch around loop for stripmining.
    NewPreheader = SplitBlock(Preheader, Preheader->getTerminator(), DT, LI);
    NewPreheader->setName(Preheader->getName() + ".new");
    // Split LatchExit to create phi nodes from branch above.
    SmallVector<BasicBlock *, 4> Preds(predecessors(LatchExit));
    NewExit = SplitBlockPredecessors(LatchExit, Preds, ".strpm-lcssa", DT, LI,
                                     nullptr, PreserveLCSSA);
    // NewExit gets its DebugLoc from LatchExit, which is not part of the
    // original Loop.
    // Fix this by setting Loop's DebugLoc to NewExit.
    auto *NewExitTerminator = NewExit->getTerminator();
    NewExitTerminator->setDebugLoc(Header->getTerminator()->getDebugLoc());
    // Split NewExit to insert epilog remainder loop.
    EpilogPreheader = SplitBlock(NewExit, NewExitTerminator, DT, LI);
    EpilogPreheader->setName(Header->getName() + ".epil.preheader");
  }

  // Calculate conditions for branch around loop for stripmining
  // in epilog case and around prolog remainder loop in prolog case.
  // Compute the number of extra iterations required, which is:
  //  extra iterations = run-time trip count % loop stripmine factor
  PreheaderBR = cast<BranchInst>(Preheader->getTerminator());

  // Loop structure should be the following:
  //  Epilog
  //
  // Preheader
  // *NewPreheader
  //   Header
  //   ...
  //   Latch
  // *NewExit
  // *EpilogPreheader
  // LatchExit

  IRBuilder<> B(PreheaderBR);
  // Int the gpu case we don't need an epilogue
  // If we start with forall(i=0..n)
  // GPU stripmine converts to
  //   forall(i=0; i<k; i++)
  //     for(j=i; j+=k; j<n)
  // CPU stripmine converts to
  //   forall(i=0; i<n/k; i++){
  //     for(j=i*k; j<(i+1)*k; j++)
  //   }
  //   epilogue

  Value *ModVal = TripCount;
  // B.SetInsertPoint(F->getEntryBlock().getFirstNonPHI());

  // FIXME: The name of this variable is infuriatingly bad!
  Instruction *bloc2;
  if (Instruction *I = dyn_cast<Instruction>(TripCount))
    bloc2 = I->getNextNode();
  else
    bloc2 = F->getEntryBlock().getTerminator();

  Value *StepSize = IRBuilder<>(bloc2).CreateCall(
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::tapir_loop_grainsize,
                                        {TripCount->getType()}),
      {TripCount});

  Value *BranchVal =
      B.CreateICmpULE(ModVal, ConstantInt::get(ModVal->getType(), 0));

  BasicBlock *RemainderLoopBB = NewExit;
  BasicBlock *StripminedLoopBB = NewPreheader;
  // Branch to either remainder (extra iterations) loop or stripmined loop.
  B.CreateCondBr(BranchVal, RemainderLoopBB, StripminedLoopBB);
  PreheaderBR->eraseFromParent();
  if (DT)
    DT->changeImmediateDominator(NewExit, Preheader);

  // Get an ordered list of blocks in the loop to help with the ordering of the
  // cloned blocks in the prolog/epilog code
  LoopBlocksDFS LoopBlocks(L);
  LoopBlocks.perform(LI);

  // Collect extra blocks in the task that LoopInfo does not consider to be part
  // of the loop, e.g., exception-handling code for the task.
  SmallVector<BasicBlock *, 8> ExtraTaskBlocks;
  SmallVector<BasicBlock *, 8> SharedEHTaskBlocks;
  SmallPtrSet<BasicBlock *, 8> SharedEHBlockPreds;
  {
    SmallPtrSet<Spindle *, 8> Visited;
    for (Task *SubT : depth_first(T)) {
      for (Spindle *S :
           depth_first<InTask<Spindle *>>(SubT->getEntrySpindle())) {
        // Only visit shared-eh spindles once a piece.
        if (S->isSharedEH() && !Visited.insert(S).second)
          continue;

        for (BasicBlock *BB : S->blocks()) {
          // Skip blocks in the loop.
          if (!L->contains(BB)) {
            ExtraTaskBlocks.push_back(BB);

            if (!T->simplyEncloses(BB) && S->isSharedEH()) {
              SharedEHTaskBlocks.push_back(BB);
              if (S->getEntry() == BB)
                for (BasicBlock *Pred : predecessors(BB))
                  if (T->simplyEncloses(Pred))
                    SharedEHBlockPreds.insert(Pred);
            }
          }
        }
      }
    }
  }

  SmallVector<Instruction *, 1> Reattaches;
  SmallVector<BasicBlock *, 4> EHBlocksToClone;
  SmallPtrSet<BasicBlock *, 4> EHBlockPreds;
  SmallPtrSet<LandingPadInst *, 1> InlinedLPads;
  SmallVector<Instruction *, 1> DetachedRethrows;
  // Analyze the original task for serialization.
  AnalyzeTaskForSerialization(T, Reattaches, EHBlocksToClone, EHBlockPreds,
                              InlinedLPads, DetachedRethrows);
  bool NeedToInsertTaskFrame = taskContainsSync(T);

  // If this detach can throw, get the exceptional continuation of the detach
  // and its associated landingpad value.
  BasicBlock *EHCont = nullptr;
  Value *EHContLPadVal = nullptr;
  SmallVector<BasicBlock *, 2> UDPreds;
  if (DI->hasUnwindDest()) {
    EHCont = T->getEHContinuationSpindle()->getEntry();
    EHContLPadVal = T->getLPadValueInEHContinuationSpindle();
    getEHContPredecessors(EHCont, T, UDPreds, *TI);
  }

  // For each extra loop iteration, create a copy of the loop's basic blocks
  // and generate a condition that branches to the copy depending on the
  // number of 'left over' iterations.
  //
  std::vector<BasicBlock *> NewBlocks;
  ValueToValueMapTy VMap;

  // TODO: For stripmine factor 2 remainder loop will have 1 iterations.
  // Do not create 1 iteration loop.
  // bool CreateRemainderLoop = (Count != 2);
  bool CreateRemainderLoop = !GPU;

  // Clone all the basic blocks in the loop. If Count is 2, we don't clone
  // the loop, otherwise we create a cloned loop to execute the extra
  // iterations. This function adds the appropriate CFG connections.
  BasicBlock *InsertBot = LatchExit;
  BasicBlock *InsertTop = EpilogPreheader;
  if (CreateRemainderLoop) {
    *RemainderLoop = cloneLoopBlocks(
        L, ModVal, CreateRemainderLoop, true, UnrollRemainder, InsertTop,
        InsertBot, NewPreheader, NewBlocks, LoopBlocks, ExtraTaskBlocks,
        SharedEHTaskBlocks, VMap, DT, LI, Count);

    // Insert the cloned blocks into the function.
    F->splice(InsertBot->getIterator(), &*F, NewBlocks[0]->getIterator(),
              F->end());

    // Loop structure should be the following:
    //  Epilog
    //
    // Preheader
    // NewPreheader
    //   Header
    //   ...
    //   Latch
    // NewExit
    // EpilogPreheader
    //   EpilogHeader
    //   ...
    //   EpilogLatch
    // LatchExit

    // Rewrite the cloned instruction operands to use the values created when
    // the clone is created.
    for (BasicBlock *BB : NewBlocks)
      for (Instruction &I : *BB)
        RemapInstruction(&I, VMap,
                         RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);

    // Serialize the cloned loop body to render the inner loop serial.
    {
      // Translate all the analysis for the new cloned task.
      SmallVector<Instruction *, 1> ClonedReattaches;
      for (Instruction *I : Reattaches)
        ClonedReattaches.push_back(cast<Instruction>(VMap[I]));
      SmallPtrSet<BasicBlock *, 4> ClonedEHBlockPreds;
      for (BasicBlock *B : EHBlockPreds)
        ClonedEHBlockPreds.insert(cast<BasicBlock>(VMap[B]));
      SmallVector<BasicBlock *, 4> ClonedEHBlocks;
      for (BasicBlock *B : EHBlocksToClone)
        ClonedEHBlocks.push_back(cast<BasicBlock>(VMap[B]));
      // Landing pads and detached-rethrow instructions may or may not have been
      // cloned.
      SmallPtrSet<LandingPadInst *, 1> ClonedInlinedLPads;
      for (LandingPadInst *LPad : InlinedLPads) {
        if (VMap[LPad])
          ClonedInlinedLPads.insert(cast<LandingPadInst>(VMap[LPad]));
        else
          ClonedInlinedLPads.insert(LPad);
      }
      SmallVector<Instruction *, 1> ClonedDetachedRethrows;
      for (Instruction *DR : DetachedRethrows) {
        if (VMap[DR])
          ClonedDetachedRethrows.push_back(cast<Instruction>(VMap[DR]));
        else
          ClonedDetachedRethrows.push_back(DR);
      }
      DetachInst *ClonedDI = cast<DetachInst>(VMap[DI]);
      // Serialize the new task.
      SerializeDetach(ClonedDI, ParentEntry, EHCont, EHContLPadVal,
                      ClonedReattaches, &ClonedEHBlocks, &ClonedEHBlockPreds,
                      &ClonedInlinedLPads, &ClonedDetachedRethrows,
                      NeedToInsertTaskFrame, DT, nullptr, LI);
    }
  }

  // Detach the stripmined loop.
  Value *SyncReg = DI->getSyncRegion();
  Value *NewSyncReg = SyncReg;
  BasicBlock *LoopReattach = NewExit;
  BasicBlock *LoopDetEntry = NewPreheader;

  // Get the set of new loop blocks
  SetVector<BasicBlock *> NewLoopBlocks;
  {
    LoopBlocksDFS NewLoopBlocksDFS(L);
    NewLoopBlocksDFS.perform(LI);
    LoopBlocksDFS::RPOIterator BlockBegin = NewLoopBlocksDFS.beginRPO();
    LoopBlocksDFS::RPOIterator BlockEnd = NewLoopBlocksDFS.endRPO();
    for (LoopBlocksDFS::RPOIterator BB = BlockBegin; BB != BlockEnd; ++BB)
      NewLoopBlocks.insert(*BB);
  }
  // Create structure in LI for new loop.
  Loop *ParentLoop = L->getParentLoop();
  Loop *NewLoop = LI->AllocateLoop();
  if (ParentLoop)
    ParentLoop->replaceChildLoopWith(L, NewLoop);
  else
    LI->changeTopLevelLoop(L, NewLoop);
  NewLoop->addChildLoop(L);

  // Move the detach/reattach instructions to surround the stripmined loop.
  BasicBlock *NewHeader;
  {
    SmallVector<BasicBlock *, 4> HeaderPreds;
    for (BasicBlock *Pred : predecessors(Header))
      if (Pred != Latch)
        HeaderPreds.push_back(Pred);
    NewHeader = SplitBlockPredecessors(Header, HeaderPreds, ".strpm.outer", DT,
                                       LI, nullptr, PreserveLCSSA);
  }
  BasicBlock *NewEntry =
      SplitBlock(NewHeader, NewHeader->getTerminator(), DT, LI);
  NewEntry->setName(TaskEntry->getName() + ".strpm.outer");
  SmallVector<BasicBlock *, 1> LoopReattachPreds{Latch};
  BasicBlock *NewReattB = SplitBlockPredecessors(
      LoopReattach, LoopReattachPreds, "", DT, LI, nullptr, PreserveLCSSA);
  NewReattB->setName(Latch->getName() + ".reattach");
  BasicBlock *NewLatch =
      SplitBlock(NewReattB, NewReattB->getTerminator(), DT, LI);
  NewLatch->setName(Latch->getName() + ".strpm.outer");

  // Move static allocas from TaskEntry into NewEntry.
  MoveStaticAllocasInBlock(NewEntry, TaskEntry, Reattaches);

  // Insert a new detach instruction
  BasicBlock *OrigUnwindDest = DI->getUnwindDest();
  if (OrigUnwindDest) {
    ReplaceInstWithInst(
        NewHeader->getTerminator(),
        DetachInst::Create(NewEntry, NewLatch, OrigUnwindDest, NewSyncReg));
    // Update the PHI nodes in the unwind destination of the detach.
    for (PHINode &PN : OrigUnwindDest->phis())
      PN.setIncomingBlock(PN.getBasicBlockIndex(Header), NewHeader);

    // Update DT.  Walk the path of unique successors from the unwind
    // destination to change the immediate dominators of these nodes.  Continue
    // updating until OrigDUBB equals the exceptional continuation or, as in the
    // case of a parallel epilog, we reach a detached-rethrow.
    BasicBlock *OrigDUBB = OrigUnwindDest;
    BasicBlock *NewDomCandidate = NewHeader;
    if (ParallelEpilog && NeedNestedSync)
      // We will insert a sync.unwind to OrigUnwindDest, which changes the
      // dominator.
      NewDomCandidate = DT->findNearestCommonDominator(NewHeader, LoopReattach);
    while (OrigDUBB && (OrigDUBB != EHCont)) {
      BasicBlock *OldIDom = DT->getNode(OrigDUBB)->getIDom()->getBlock();
      DT->changeImmediateDominator(
          OrigDUBB, DT->findNearestCommonDominator(OldIDom, NewDomCandidate));
      // Get the next block along the path.  If we reach the end of the path at
      // a detached-rethrow, then getUniqueSuccessor() returns nullptr.
      OrigDUBB = OrigDUBB->getUniqueSuccessor();
    }
    // If OrigDUBB equals EHCont, then this is the last block we aim to update.
    if (OrigDUBB == EHCont) {
      BasicBlock *OldIDom = DT->getNode(EHCont)->getIDom()->getBlock();
      DT->changeImmediateDominator(
          EHCont, DT->findNearestCommonDominator(OldIDom, NewDomCandidate));
    }
  } else {
    ReplaceInstWithInst(NewHeader->getTerminator(),
                        DetachInst::Create(NewEntry, NewLatch, NewSyncReg));
  }

  // Replace the old detach instruction with a branch
  ReplaceInstWithInst(Header->getTerminator(),
                      BranchInst::Create(DI->getDetached()));

  // Replace the old reattach instructions with branches.  Along the way,
  // determine their common dominator.
  BasicBlock *ReattachDom = nullptr;
  for (Instruction *I : Reattaches) {
    if (!ReattachDom)
      ReattachDom = I->getParent();
    else
      ReattachDom = DT->findNearestCommonDominator(ReattachDom, I->getParent());
    ReplaceInstWithInst(I, BranchInst::Create(Latch));
  }
  assert(ReattachDom && "No reattach-dominator block found");
  // Insert a reattach at the end of NewReattB.
  ReplaceInstWithInst(NewReattB->getTerminator(),
                      ReattachInst::Create(NewLatch, NewSyncReg));
  // Update the dominator tree, and determine predecessors of epilog.
  if (DT->dominates(Header, Latch))
    DT->changeImmediateDominator(Latch, ReattachDom);

  // The block structure of the stripmined loop should now look like so:
  //
  // LoopDetEntry
  // NewHeader (detach NewEntry, NewLatch)
  // NewEntry
  // Header
  // TaskEntry
  // ...
  // Latch (br Header, NewReattB)
  // NewReattB (reattach NewLatch)
  // NewLatch (br LoopReattach)
  // LoopReattach

  // Add check of stripmined loop count.
  IRBuilder<> B2(LoopDetEntry->getTerminator());

  // We compute the loop count of the outer loop using a UDiv by the power-of-2
  // count to ensure that ScalarEvolution can handle this outer loop once we're
  // done.
  //
  // TODO: Generalize to handle non-power-of-2 counts.
  assert(isPowerOf2_32(Count) && "Count is not a power of 2.");
  Value *TestVal = StepSize;
  PHINode *NewIdx = PHINode::Create(TestVal->getType(), 2, "niter",
                                    NewHeader->getFirstNonPHIIt());
  B2.SetInsertPoint(NewLatch->getTerminator());
  Instruction *IdxAdd = cast<Instruction>(
      B2.CreateAdd(NewIdx, ConstantInt::get(NewIdx->getType(), 1),
                   NewIdx->getName() + ".nadd"));
  IdxAdd->copyIRFlags(PrimaryInc);
  NewIdx->addIncoming(ConstantInt::get(TestVal->getType(), 0), LoopDetEntry);
  NewIdx->addIncoming(IdxAdd, NewLatch);
  Value *IdxCmp = B2.CreateICmpEQ(IdxAdd, TestVal, NewIdx->getName() + ".ncmp");
  ReplaceInstWithInst(NewLatch->getTerminator(),
                      BranchInst::Create(LoopReattach, NewHeader, IdxCmp));

  DT->changeImmediateDominator(NewLatch, NewHeader);

  // The block structure of the stripmined loop should now look like so:
  //
  // LoopDetEntry
  // NewHeader (detach NewEntry, NewLatch)
  // NewEntry
  // Header
  // TaskEntry
  // ...
  // Latch (br Header, NewReattB)
  // NewReattB (reattach NewLatch)
  // NewLatch (br NewHeader, LoopReattach)
  // LoopReattach

  // Fixup the LoopInfo for the new loop.
  if (!ParentLoop) {
    NewLoop->addBasicBlockToLoop(NewHeader, *LI);
    NewLoop->addBasicBlockToLoop(NewEntry, *LI);
    for (BasicBlock *BB : NewLoopBlocks) {
      NewLoop->addBlockEntry(BB);
    }
    NewLoop->addBasicBlockToLoop(NewReattB, *LI);
    NewLoop->addBasicBlockToLoop(NewLatch, *LI);
  } else {
    LI->changeLoopFor(NewHeader, NewLoop);
    NewLoop->addBlockEntry(NewHeader);
    LI->changeLoopFor(NewEntry, NewLoop);
    NewLoop->addBlockEntry(NewEntry);
    for (BasicBlock *BB : NewLoopBlocks)
      NewLoop->addBlockEntry(BB);
    LI->changeLoopFor(NewReattB, NewLoop);
    NewLoop->addBlockEntry(NewReattB);
    LI->changeLoopFor(NewLatch, NewLoop);
    NewLoop->addBlockEntry(NewLatch);
  }

  // Update loop metadata
  NewLoop->setLoopID(L->getLoopID());

  llvm_unreachable("NOT IMPLEMENTED: Clearing loop hints metadata");
  // TapirLoopHints Hints(L);
  // Hints.clearHintsMetadata();

  // Update all of the old PHI nodes
  B2.SetInsertPoint(NewEntry->getTerminator());

  // GPU mode: inner loop strides by grainsize.
  PHINode *InnerIdx = PHINode::Create(PrimaryInduction->getType(), 2,
                                      "inneriter", Header->getFirstNonPHIIt());
  // Initialize inner index to zero.
  // Value *Zero = ConstantInt::get(PrimaryInduction->getType(), 0);
  B2.SetInsertPoint(LatchBR->getParent()->getFirstNonPHIIt());
  // Instead of subtracting one, add the grainsize.

  Value *NextIdx =
      B2.CreateAdd(InnerIdx, StepSize, InnerIdx->getName() + ".nadd_stride");

  // NextIdx->copyIRFlags(PrimaryInc);
  //  Check if the new index is still within the original trip count.
  InnerIdx->addIncoming(NewIdx, NewEntry);
  InnerIdx->addIncoming(NextIdx, Latch);
  Value *InnerCmp;
  if (LatchBR->getSuccessor(0) == Header)
    InnerCmp = B2.CreateICmpULT(NextIdx, TripCount,
                                InnerIdx->getName() + ".ncmp_final");

  else
    InnerCmp = B2.CreateICmpUGE(NextIdx, TripCount,
                                InnerIdx->getName() + ".ncmp_final");

  LatchBR->setCondition(InnerCmp);
  // In the gpu case, we actually want to replace the induction variable
  PrimaryInduction->replaceAllUsesWith(InnerIdx);

  // Connect the epilog code to the original loop and update the PHI functions.
  B2.SetInsertPoint(EpilogPreheader->getTerminator());

  // If this loop is nested, then the loop stripminer changes the code in the
  // any of its parent loops, so the Scalar Evolution pass needs to be run
  // again.
  SE->forgetTopmostLoop(L);

  // At this point, the code is well formed.  We now simplify the new loops,
  // doing constant propagation and dead code elimination as we go.
  simplifyLoopAfterStripMine(L, /*SimplifyIVs*/ true, LI, SE, DT, TTI, AC);
  simplifyLoopAfterStripMine(NewLoop, /*SimplifyIVs*/ true, LI, SE, DT, TTI,
                             AC);
  if (!GPU)
    simplifyLoopAfterStripMine(*RemainderLoop, /*SimplifyIVs*/ true, LI, SE, DT,
                               TTI, AC);

  // TODO: update all the analyses manually
#ifndef NDEBUG
  // DT->verify();
  // LI->verify(*DT);
#endif

  // Record that the old loop was derived from a Tapir loop.
  L->setDerivedFromTapirLoop();

  // Update TaskInfo manually using the updated DT.
  // if (TI)
  // FIXME: Recalculating TaskInfo for the whole function is wasteful.
  // Optimize this routine in the future.
  // TI->recalculate(*F, *DT);

  // Reductions take a parallel loop
  // forall(i=0; i<n; i++)
  //   Can be an aribrary parallel loop with multiple reductions
  //   BODY
  //   sum(&red, a[i]; 0.0)
  //
  // GPU stripmine converts to
  //   nred = gpuGridSize(n);
  //   reds = managedMalloc(nred);
  //   forall(i=0; i<nred; i++){
  //     for(j=i; j<n; j+=nred) {
  //       BODY
  //       sum(&reds[i], a[j]; 0.0)
  //     }
  //   }
  //   for(i=0; i<nred; i++)
  //     sum(&red, reds[i], 0.0);
  //
  // CPU stripmine converts to
  //   nred = n/K; // K defaults to 2048 (DefaultCoarseningFactor)
  //   reds = managedMalloc();
  //   forall(i=0; i<n; i+=nred){ // not quite right, need to handle case with
  //   epilogue
  //     reds[i] = 0.0
  //     for(j=i; j<i+nred; j++){
  //       BODY
  //       sum(&reds[i], a[j]; 0.0)
  //     }
  //   }
  //   // Epilogue (leftover iterations)
  //   if(nred * k > N){
  //     for(...) // epilogue logic
  //       BODY
  //       sum(&reds[i], a[j]; 0.0)
  //   }
  //   for(i=0; i<nred; i++)
  //     sum(&red, reds[i], 0.0);
  //
  // record calls to reduction functions in loop for later reference

  const std::vector<BasicBlock *> &blocks = L->getBlocks();
  std::set<std::pair<CallInst *, Type *>> reductions;
  for (BasicBlock *BB : blocks) {
    for (Instruction &I : *BB) {
      if (auto ci = dyn_cast<CallInst>(&I)) {
        auto f = ci->getCalledFunction();
        if (isReducer(*f)) {
          LLVM_DEBUG(dbgs()
                     << "Found reduction var: "
                     << ci->getArgOperand(0)->getName()
                     << "with reduction function: " << f->getName() << "\n");
          auto ty = ci->getArgOperand(1)->getType();
          reductions.insert(std::make_pair(ci, ty));
          // TODO: check the type to confirm valid reduction
        }
      }
    }
  }

  // To make the stripmining work for multiple backends, we parameterize on the
  // step and the termination condition Roughly speaking, we want the CPU to
  // look like
  // n = p*s + k
  // forall(i = 0; i<p; j++)
  //   for(j = i*s; j < (i+1)*s, j++)
  //     B
  //
  // forall(i = p*s; i<n; i++)
  //   B
  //
  // while GPU should look like
  // forall(i=0..p)
  //   for(j = 0; j < n; j+=s)
  //     B
  //
  //
  // accumulate reductions in epilog loop
  LLVM_DEBUG(dbgs() << "Found " << reductions.size()
                    << " reduction variables in loop\n");

  // Associates calls to reduction functions, first argument to reduction
  // function,  local reduction allocation, type of unit, unit
  std::vector<std::tuple<CallInst *, Value *, Value *, Type *, Value *>> redMap;
  // TODO: Modify the strip mining outer loop to be smaller: currently we are
  // stack allocating n/2048 reduction values.
  // TODO: Initialize local reductions with unit values
  // TODO: move insertion point for reduction allocation
  // TODO: free reduction allocation
  Instruction *bloc = nullptr;
  if (Instruction *I = dyn_cast<Instruction>(TripCount)) {
    bloc = I->getParent()->getTerminator();
  } else {
    bloc = F->getEntryBlock().getTerminator();
  }
  IRBuilder<> RB(bloc);
  Value *outerIters = StepSize;
  const DataLayout &DL = Header->getModule()->getDataLayout();

  // Here we iterate over the reductions (calls to reduction functions), and
  // allocate the local reduction variable array, and build the association
  // array redMap, and replace references to the original reduction variable
  // with references to the new local reduction variable in the body of the
  // inner loop
  Value *nred =
      RB.CreateAdd(outerIters, ConstantInt::get(outerIters->getType(), 1));
  for (auto &pair : reductions) {
    // TODO: generic allocation/free calls
    CallInst *ci = pair.first;
    Value *ptr = ci->getArgOperand(0);
    Value *unit = ci->getArgOperand(2);
    Type *ty = pair.second;
    FunctionType *gmmTy =
        FunctionType::get(ptr->getType(), {nred->getType()}, false);
    Value *arrSize = RB.CreateMul(
        nred, ConstantInt::get(nred->getType(), DL.getTypeAllocSize(ty)));
    Value *al = RB.CreateCall(M->getOrInsertFunction("gpuManagedMalloc", gmmTy),
                              {arrSize});
    // auto al = RB.CreateBitCast(rm, ty);
    // auto al = RB.CreateAlloca(ty, nred, ptr->getName() + "_reduction");
    IRBuilder<> BH(NewLoop->getHeader()->getTerminator());
    Value *lptr =
        BH.CreateBitCast(BH.CreateGEP(ty, al, NewIdx), ptr->getType());
    redMap.push_back(std::make_tuple(ci, ptr, al, ty, unit));
    // Assume there is more than one element, and
    // use the first element for the first iteration of the loop.
    // roughly:
    //   red = init;
    //   forall(i = ...){
    //     red = reduce(red, body(i));
    //   }
    //   red = init;
    //   localred[m+1];
    //
    //   forall(k ∈ 0..m-1){
    //     localred[i] = body(j_0);
    //     for(j ∈ j_k_1..j_k_l-1)
    //       reduce(localred+i, body(j));
    //   }
    //   for( j ∈ j_k_m .. n )
    //     reduce(localred+m, body(j));
    //
    //   for(k ∈ 0..m)
    //     reduce(&red, localred[k]);
    //
    ptr->replaceUsesWithIf(lptr, [L](Use &u) {
      if (auto I = dyn_cast<Instruction>(u.getUser()))
        return L->contains(I->getParent());
      else
        return false;
    });
  }

  // Epilog "join" of reduction values stored in local reduction value arrays.
  // Should be able to use redMap to map original pointer (which is still used
  // to reduce the remainder of the strimined loop, so you probably want to
  // start the reduction with that value).
  LLVM_DEBUG(
      dbgs() << "Function after strip mining, before reduction epilogue\n"
             << *F);

  if (!reductions.empty()) {
    ValueToValueMapTy VMap;
    SmallVector<Instruction *, 4> CIS;
    for (BasicBlock *BB : NewLoop->blocks()) {
      // We find the location that we reduce into and create a store of unit
      // TODO: Get unit value for reduction
      for (Instruction &I : *BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          Function *F = CI->getCalledFunction();
          if (isReducer(*F)) {
            // this must be defined in the outer parallel loop but before the
            // inner loop
            IRBuilder<> PB(
                dyn_cast<Instruction>(CI->getArgOperand(0))->getNextNode());
            PB.CreateStore(CI->getArgOperand(2), CI->getArgOperand(0));
            CIS.push_back(&I);
            F->removeFnAttr(Attribute::NoInline);
            F->removeFnAttr(Attribute::OptimizeNone);
          }
        }
      }
    }

    // We insert the reduction code at every sync corresponding to the strimined
    // loop
    //
    // Sync
    // RedEpiHeader
    //   RedEpiBody
    // RedEpiExit

    Instruction *term = LatchExit->getTerminator();
    BasicBlock *PostSync = term->getSuccessor(0);
    BasicBlock *RedEpiHeader =
        BasicBlock::Create(LatchExit->getContext(), "reductionEpilogue",
                           LatchExit->getParent(), LatchExit);
    RedEpiHeader->moveAfter(LatchExit);
    ReplaceInstWithInst(term, SyncInst::Create(RedEpiHeader, SyncReg));
    BranchInst::Create(PostSync, RedEpiHeader);
    PHINode *Idx =
        PHINode::Create(outerIters->getType(), 2, "reductionepilogueidx",
                        RedEpiHeader->getFirstNonPHIIt());
    IRBuilder<> BH(RedEpiHeader, RedEpiHeader->getFirstNonPHIIt());
    Idx->addIncoming(ConstantInt::get(outerIters->getType(), 0), LatchExit);
    Instruction *bodyTerm, *exitTerm;
    Value *cmp = BH.CreateCmp(CmpInst::ICMP_NE, Idx, outerIters);
    SplitBlockAndInsertIfThenElse(cmp, RedEpiHeader->getTerminator(), &bodyTerm,
                                  &exitTerm);

    IRBuilder<> BB(bodyTerm);
    // For each reduction, get the allocated thread local reduced values and
    // reduce them.
    for (auto &kv : redMap) {
      const auto [ci, ptr, al, ty, unit] = kv;
      Value *lptr = BB.CreateBitCast(BB.CreateGEP(ty, al, Idx), ptr->getType());
      Value *x = BB.CreateLoad(ty, lptr);
      BB.SetCurrentDebugLocation(ci->getDebugLoc());
      BB.CreateCall(ci->getCalledFunction(), {ptr, x, unit});
    }
    Value *IdxAdd = BB.CreateAdd(Idx, ConstantInt::get(Idx->getType(), 1),
                                 Idx->getName() + ".add");
    BasicBlock *body = bodyTerm->getParent();
    Idx->addIncoming(IdxAdd, body);
    ReplaceInstWithInst(bodyTerm, BranchInst::Create(RedEpiHeader));

    // Update Loopinfo with reduction loop
    Loop *RL = LI->AllocateLoop();
    if (ParentLoop)
      ParentLoop->addChildLoop(RL);
    else
      LI->addTopLevelLoop(RL);
    if (!ParentLoop) {
      RL->addBasicBlockToLoop(RedEpiHeader, *LI);
      RL->addBasicBlockToLoop(body, *LI);
    } else {
      LI->changeLoopFor(RedEpiHeader, RL);
      RL->addBlockEntry(RedEpiHeader);
      LI->changeLoopFor(body, RL);
      RL->addBlockEntry(body);
    }
  }

  LLVM_DEBUG(dbgs() << "Function after reduction epilogue\n" << *F);

  // TODO: fix DT updates
  // DT->recalculate(*F);

#ifndef NDEBUG
  // DT->verify();
  // LI->verify(*DT);
#endif

  return NewLoop;
}

static bool tryToStripMineReductionLoop(
    Loop *loop, DominatorTree &dt, LoopInfo *li, ScalarEvolution &se,
    const TargetTransformInfo &tti, AssumptionCache &ac, TaskInfo *ti,
    OptimizationRemarkEmitter &ore, TargetLibraryInfo *tli, bool preserveLCSSA,
    std::optional<unsigned> providedCount) {
  // TODO: Enable this assertion. It requires a loop attribute indicating that
  // the loop is a reduction loop.
  // assert(isReductionLoop(*loop) && "Must be a reduction loop");
  assert(hasTargetAttr(*loop) && "Reduction loop must be a tapir loop");
  assert(loop->isLoopSimplifyForm() && "Loop must be in loop-simplify form");

  SmallPtrSet<const Value *, 32> ephValues;
  CodeMetrics::collectEphemeralValues(loop, &ac, ephValues);

  WSCost loopCost;
  estimateLoopCost(loopCost, loop, li, &se, tti, tli, ephValues);

  // FIXME: Instead of asserting, we should probably raise an error.
  assert(!loopCost.Metrics.notDuplicatable &&
         "Tapir reduction loop cannot contain non-duplicatable instructions");
  assert(loopCost.Metrics.Convergence == ConvergenceKind::None &&
         "Tapir reduction loop cannot contain convergent operations");

  // Save loop properties before it is transformed.
  MDNode *origLoopID = loop->getLoopID();
  Loop *remainderLoop = nullptr;
  Loop *newLoop = StripMineReductionLoop(
      loop, /*Count=*/0, /*AllowExpensiveTripCount=*/true,
      /*UnrollRemainder=*/false, li, &se, &dt, tti, &ac, ti, &ore,
      preserveLCSSA, /*ParallelEpilog=*/false,
      /*NeedNestedSync*/ true, &remainderLoop, /*GPU=*/true);

  // TODO: If we cannot strip-mine the loop, we should probably fail
  // catastrophically because we will not be able to carry out a parallel
  // reduction on the GPU in that case.
  if (!newLoop)
    return false;

  // Copy metadata to remainder loop
  if (remainderLoop && origLoopID) {
    MDNode *newRemainderLoopID =
        CopyNonTapirLoopMetadata(remainderLoop->getLoopID(), origLoopID);
    remainderLoop->setLoopID(newRemainderLoopID);
  }

  // Mark the new loop as processed for reductions.
  errs() << "\n";
  errs() << "----------------------------------------------------------\n";
  errs() << "WARNING: Reduction loop not marked as processed\n";
  errs() << "----------------------------------------------------------\n";
  errs() << "\n";

  // This should require some attribute to be defined.
  llvm_unreachable("Annotate reduction loop as having been processed");
  // setReductionLoopProcessed(loop);

  return true;
}

static bool isReductionLoop(const Loop &loop) {
  errs() << "\n";
  errs() << "----------------------------------------------------------\n";
  errs() << "WARNING: Reduction loop attribute has not been implemented\n";
  errs() << "         Assuming all loops are reduction loops\n";
  errs() << "          Yes, this is OBVIOUSLY TERRIBLE\n";
  errs() << "----------------------------------------------------------\n";
  errs() << "\n";

  // Obviously, this should not always return true.
  return true;
}

PreservedAnalyses
PrepareReductionLoopsGPUPass::run(Function &f, FunctionAnalysisManager &am) {
  TargetLibraryInfo &tli = am.getResult<TargetLibraryAnalysis>(f);
  ScalarEvolution &se = am.getResult<ScalarEvolutionAnalysis>(f);
  LoopInfo &li = am.getResult<LoopAnalysis>(f);
  TargetTransformInfo &tti = am.getResult<TargetIRAnalysis>(f);
  DominatorTree &dt = am.getResult<DominatorTreeAnalysis>(f);
  AssumptionCache &ac = am.getResult<AssumptionAnalysis>(f);
  TaskInfo &ti = am.getResult<TaskAnalysis>(f);
  OptimizationRemarkEmitter &ore =
      am.getResult<OptimizationRemarkEmitterAnalysis>(f);

  LoopAnalysisManager *lam = nullptr;
  if (auto *lamProxy = am.getCachedResult<LoopAnalysisManagerFunctionProxy>(f))
    lam = &lamProxy->getManager();

  bool changed = false;

  for (Loop *loop : li) {
    changed |= simplifyLoop(loop, &dt, &li, &se, &ac, nullptr,
                            /* PreserveLCSSA */ false);
    changed |= formLCSSARecursively(*loop, dt, &li, &se);
  }

  SmallPriorityWorklist<Loop *, 4> wl;
  appendLoopsToWorklist(li, wl);

  while (!wl.empty()) {
    // Because the LoopInfo stores the loops in RPO, we walk the worklist from
    // back to front so that we work forward across the CFG, which for
    // stripmining is only needed to get optimization remarks emitted in a
    // forward order.
    Loop &loop = *wl.pop_back_val();

    // FIXME: We should use a loop attribute to identify loops that perform
    // reductions. This has not been implemented yet, so isReductionLoop will
    // always return true. At some point, when support for this attribute is
    // added, this should be replaced with the a direct lookup of the
    // attribute.
    if (!hasTargetAttr(loop) || !isGPUTT(*getTargetAttr(loop)) ||
        !isReductionLoop(loop))
      continue;
#ifndef NDEBUG
    Loop *parentLoop = loop.getParentLoop();
#endif

    std::string loopName = std::string(loop.getName());
    bool loopChanged = tryToStripMineReductionLoop(
        &loop, dt, &li, se, tti, ac, &ti, ore, &tli,
        /*PreserveLCSSA*/ true, /*Count*/ std::nullopt);
    changed |= loopChanged;

    // The parent must not be damaged by stripmining!
#ifndef NDEBUG
    parentLoop->verifyLoop();
#endif

    // Clear any cached analysis results for loop if we removed it completely.
    if (lam && loopChanged)
      lam->clear(loop, loopName);
  }

  if (!changed)
    return PreservedAnalyses::all();
  return getLoopPassPreservedAnalyses();
}
