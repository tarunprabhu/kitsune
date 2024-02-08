//===--- CGCilk.cpp - Emit LLVM Code for Cilk expressions -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with code generation of Cilk statements and
// expressions.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CGCleanup.h"

using namespace clang;
using namespace CodeGen;

CodeGenFunction::IsSpawnedScope::IsSpawnedScope(CodeGenFunction *CGF)
    : CGF(CGF), OldIsSpawned(CGF->IsSpawned),
      OldSpawnedCleanup(CGF->SpawnedCleanup) {
  CGF->IsSpawned = false;
  CGF->SpawnedCleanup = OldIsSpawned;
}

CodeGenFunction::IsSpawnedScope::~IsSpawnedScope() {
  RestoreOldScope();
}

bool CodeGenFunction::IsSpawnedScope::OldScopeIsSpawned() const {
  return OldIsSpawned;
}

void CodeGenFunction::IsSpawnedScope::RestoreOldScope() {
  CGF->IsSpawned = OldIsSpawned;
  CGF->SpawnedCleanup = OldSpawnedCleanup;
}

void CodeGenFunction::EmitImplicitSyncCleanup(llvm::Instruction *SyncRegion) {
  llvm::Instruction *SR = SyncRegion;
  // If a sync region wasn't specified with this cleanup initially, try to grab
  // the current sync region.
  if (!SR && CurSyncRegion && CurSyncRegion->getSyncRegionStart())
    SR = CurSyncRegion->getSyncRegionStart();
  if (!SR)
    return;

  llvm::BasicBlock *ContinueBlock = createBasicBlock("sync.continue");
  Builder.CreateSync(ContinueBlock, SR);
  EmitBlockAfterUses(ContinueBlock);
  if (getLangOpts().Exceptions && !CurFn->doesNotThrow())
    EmitCallOrInvoke(CGM.getIntrinsic(llvm::Intrinsic::sync_unwind), { SR });
}

void CodeGenFunction::DetachScope::CreateTaskFrameEHState() {
  // Save the old EH state.
  OldEHResumeBlock = CGF.EHResumeBlock;
  CGF.EHResumeBlock = nullptr;
  OldExceptionSlot = CGF.ExceptionSlot;
  CGF.ExceptionSlot = nullptr;
  OldEHSelectorSlot = CGF.EHSelectorSlot;
  CGF.EHSelectorSlot = nullptr;
  OldNormalCleanupDest = CGF.NormalCleanupDest;
  CGF.NormalCleanupDest = Address::invalid();
}

void CodeGenFunction::DetachScope::CreateDetachedEHState() {
  // Save the old EH state.
  TFEHResumeBlock = CGF.EHResumeBlock;
  CGF.EHResumeBlock = nullptr;
  TFExceptionSlot = CGF.ExceptionSlot;
  CGF.ExceptionSlot = nullptr;
  TFEHSelectorSlot = CGF.EHSelectorSlot;
  CGF.EHSelectorSlot = nullptr;
  TFNormalCleanupDest = CGF.NormalCleanupDest;
  CGF.NormalCleanupDest = Address::invalid();
}

llvm::BasicBlock *CodeGenFunction::DetachScope::RestoreTaskFrameEHState() {
  llvm::BasicBlock *NestedEHResumeBlock = CGF.EHResumeBlock;
  CGF.EHResumeBlock = TFEHResumeBlock;
  CGF.ExceptionSlot = TFExceptionSlot;
  CGF.EHSelectorSlot = TFEHSelectorSlot;
  CGF.NormalCleanupDest = TFNormalCleanupDest;
  return NestedEHResumeBlock;
}

llvm::BasicBlock *CodeGenFunction::DetachScope::RestoreParentEHState() {
  llvm::BasicBlock *NestedEHResumeBlock = CGF.EHResumeBlock;
  CGF.EHResumeBlock = OldEHResumeBlock;
  CGF.ExceptionSlot = OldExceptionSlot;
  CGF.EHSelectorSlot = OldEHSelectorSlot;
  CGF.NormalCleanupDest = OldNormalCleanupDest;
  return NestedEHResumeBlock;
}

void CodeGenFunction::DetachScope::EnsureTaskFrame() {
  if (!TaskFrame) {
    llvm::Function *TaskFrameCreate =
        CGF.CGM.getIntrinsic(llvm::Intrinsic::taskframe_create);
    TaskFrame = CGF.Builder.CreateCall(TaskFrameCreate);

    // Create a new alloca insertion point within the task frame.
    OldAllocaInsertPt = CGF.AllocaInsertPt;
    llvm::Value *Undef = llvm::UndefValue::get(CGF.Int32Ty);
    CGF.AllocaInsertPt = new llvm::BitCastInst(Undef, CGF.Int32Ty, "",
                                               CGF.Builder.GetInsertBlock());
    // SavedDetachedAllocaInsertPt = CGF.AllocaInsertPt;

    CreateTaskFrameEHState();

    CGF.pushFullExprCleanup<CallTaskEnd>(
        static_cast<CleanupKind>(EHCleanup | LifetimeMarker | TaskExit),
        TaskFrame);
  }
}

void CodeGenFunction::DetachScope::InitDetachScope() {
  // Create the detached and continue blocks.
  DetachedBlock = CGF.createBasicBlock("det.achd");
  ContinueBlock = CGF.createBasicBlock("det.cont");
}

void CodeGenFunction::DetachScope::PushSpawnedTaskTerminate() {
  CGF.pushFullExprCleanupImpl<CallDetRethrow>(
      // This cleanup should not be a TaskExit, because we've pushed a TaskExit
      // cleanup onto EHStack already, corresponding with the taskframe.
      static_cast<CleanupKind>(EHCleanup | LifetimeMarker),
      CGF.CurSyncRegion->getSyncRegionStart());
}

void CodeGenFunction::DetachScope::StartDetach() {
  InitDetachScope();

  // Set the detached block as the new alloca insertion point.
  TFAllocaInsertPt = CGF.AllocaInsertPt;
  llvm::Value *Undef = llvm::UndefValue::get(CGF.Int32Ty);
  CGF.AllocaInsertPt = new llvm::BitCastInst(Undef, CGF.Int32Ty, "",
                                             DetachedBlock);

  if (StmtCleanupsScope)
    StmtCleanupsScope->DoDetach();
  else
    PushSpawnedTaskTerminate();

  // Create the detach
  Detach = CGF.Builder.CreateDetach(DetachedBlock, ContinueBlock,
                                    CGF.CurSyncRegion->getSyncRegionStart());

  // Save the old EH state.
  CreateDetachedEHState();

  // Emit the detached block.
  CGF.EmitBlock(DetachedBlock);

  // Link this detach block to the task frame, if it exists.
  if (TaskFrame) {
    llvm::Function *TaskFrameUse =
        CGF.CGM.getIntrinsic(llvm::Intrinsic::taskframe_use);
    CGF.Builder.CreateCall(TaskFrameUse, { TaskFrame });
  }

  // For Cilk, ensure that the detached task is implicitly synced before it
  // returns.
  CGF.PushSyncRegion()->addImplicitSync();

  // Initialize lifetime intrinsics for the reference temporary.
  if (RefTmp.isValid()) {
    switch (RefTmpSD) {
    case SD_Automatic:
    case SD_FullExpression:
      if (auto *Size = CGF.EmitLifetimeStart(
              CGF.CGM.getDataLayout().getTypeAllocSize(RefTmp.getElementType()),
              RefTmp.getPointer())) {
        if (RefTmpSD == SD_Automatic)
          CGF.pushCleanupAfterFullExpr<CallLifetimeEnd>(NormalEHLifetimeMarker,
                                                        RefTmp, Size);
        else
          CGF.pushFullExprCleanup<CallLifetimeEnd>(NormalEHLifetimeMarker,
                                                   RefTmp, Size);
      }
      break;
    default:
      break;
    }
  }

  DetachStarted = true;
}

void CodeGenFunction::DetachScope::CleanupDetach() {
  if (!DetachStarted || DetachCleanedUp)
    return;

  // Pop the sync region for the detached task.
  CGF.PopSyncRegion();
  DetachCleanedUp = true;
}

void CodeGenFunction::DetachScope::EmitTaskEnd() {
  if (!CGF.HaveInsertPoint())
    return;

  // The CFG path into the spawned statement should terminate with a `reattach'.
  CGF.Builder.CreateReattach(ContinueBlock,
                             CGF.CurSyncRegion->getSyncRegionStart());
}

static void EmitTrivialLandingPad(CodeGenFunction &CGF,
                                  llvm::BasicBlock *TempInvokeDest) {
  // Save the current IR generation state.
  CGBuilderTy::InsertPoint savedIP = CGF.Builder.saveAndClearIP();

  // Insert a simple cleanup landingpad at the start of TempInvokeDest.
  TempInvokeDest->setName("lpad");
  CGF.EmitBlock(TempInvokeDest);
  CGF.Builder.SetInsertPoint(&TempInvokeDest->front());

  llvm::LandingPadInst *LPadInst =
      CGF.Builder.CreateLandingPad(llvm::StructType::get(CGF.Int8PtrTy,
                                                         CGF.Int32Ty), 0);

  llvm::Value *LPadExn = CGF.Builder.CreateExtractValue(LPadInst, 0);
  CGF.Builder.CreateStore(LPadExn, CGF.getExceptionSlot());
  llvm::Value *LPadSel = CGF.Builder.CreateExtractValue(LPadInst, 1);
  CGF.Builder.CreateStore(LPadSel, CGF.getEHSelectorSlot());

  LPadInst->setCleanup(true);

  // Restore the old IR generation state.
  CGF.Builder.restoreIP(savedIP);
}

void CodeGenFunction::DetachScope::FinishDetach() {
  if (!DetachStarted)
    return;

  CleanupDetach();
  // Pop the detached_rethrow.
  CGF.PopCleanupBlock();

  EmitTaskEnd();

  // Restore the alloca insertion point to taskframe_create.
  {
    llvm::Instruction *Ptr = CGF.AllocaInsertPt;
    CGF.AllocaInsertPt = TFAllocaInsertPt;
    SavedDetachedAllocaInsertPt = nullptr;
    Ptr->eraseFromParent();
  }

  // Restore the task frame's EH state.
  llvm::BasicBlock *TaskResumeBlock = RestoreTaskFrameEHState();
  assert(!TaskResumeBlock && "Emission of task produced a resume block");

  llvm::BasicBlock *InvokeDest = nullptr;
  if (TempInvokeDest) {
    InvokeDest = CGF.getInvokeDest();
    if (InvokeDest)
      TempInvokeDest->replaceAllUsesWith(InvokeDest);
    else {
      InvokeDest = TempInvokeDest;
      EmitTrivialLandingPad(CGF, TempInvokeDest);
      TempInvokeDest = nullptr;
    }
  }

  // Emit the continue block.
  CGF.EmitBlock(ContinueBlock);

  // If the detached-rethrow handler is used, add an unwind destination to the
  // detach.
  if (InvokeDest) {
    CGBuilderTy::InsertPoint SavedIP = CGF.Builder.saveIP();
    CGF.Builder.SetInsertPoint(Detach);
    // Create the new detach instruction.
    llvm::DetachInst *NewDetach = CGF.Builder.CreateDetach(
        Detach->getDetached(), Detach->getContinue(), InvokeDest,
        Detach->getSyncRegion());
    // Remove the old detach.
    Detach->eraseFromParent();
    Detach = NewDetach;
    CGF.Builder.restoreIP(SavedIP);
  }

  // Pop the taskframe.
  CGF.PopCleanupBlock();

  // Restore the alloca insertion point.
  {
    llvm::Instruction *Ptr = CGF.AllocaInsertPt;
    CGF.AllocaInsertPt = OldAllocaInsertPt;
    TFAllocaInsertPt = nullptr;
    Ptr->eraseFromParent();
  }

  // Restore the original EH state.
  llvm::BasicBlock *NestedEHResumeBlock = RestoreParentEHState();

  if (TempInvokeDest) {
    if (llvm::BasicBlock *InvokeDest = CGF.getInvokeDest()) {
      TempInvokeDest->replaceAllUsesWith(InvokeDest);
    } else
      EmitTrivialLandingPad(CGF, TempInvokeDest);
  }

  // If invocations in the parallel task led to the creation of EHResumeBlock,
  // we need to create for outside the task.  In particular, the new
  // EHResumeBlock must use an ExceptionSlot and EHSelectorSlot allocated
  // outside of the task.
  if (NestedEHResumeBlock) {
    if (!NestedEHResumeBlock->use_empty()) {
      // Translate the nested EHResumeBlock into an appropriate EHResumeBlock in
      // the outer scope.
      NestedEHResumeBlock->replaceAllUsesWith(
          CGF.getEHResumeBlock(
              isa<llvm::ResumeInst>(NestedEHResumeBlock->getTerminator())));
    }
    delete NestedEHResumeBlock;
  }
}

Address CodeGenFunction::DetachScope::CreateDetachedMemTemp(
    QualType Ty, StorageDuration SD, const Twine &Name) {
  // There shouldn't be multiple reference temporaries needed.
  assert(!RefTmp.isValid() &&
         "Already created a reference temporary in this detach scope.");

  // Create the reference temporary
  RefTmp = CGF.CreateMemTemp(Ty, Name);
  RefTmpSD = SD;

  return RefTmp;
}

CodeGenFunction::TaskFrameScope::TaskFrameScope(CodeGenFunction &CGF)
    : CGF(CGF) {
  if (!CGF.CurSyncRegion)
    return;

  llvm::Function *TaskFrameCreate =
      CGF.CGM.getIntrinsic(llvm::Intrinsic::taskframe_create);
  TaskFrame = CGF.Builder.CreateCall(TaskFrameCreate);

  // Create a new alloca insertion point within the task frame.
  OldAllocaInsertPt = CGF.AllocaInsertPt;
  llvm::Value *Undef = llvm::UndefValue::get(CGF.Int32Ty);
  CGF.AllocaInsertPt = new llvm::BitCastInst(Undef, CGF.Int32Ty, "",
                                             CGF.Builder.GetInsertBlock());

  // Save the old EH state.
  OldEHResumeBlock = CGF.EHResumeBlock;
  CGF.EHResumeBlock = nullptr;
  OldExceptionSlot = CGF.ExceptionSlot;
  CGF.ExceptionSlot = nullptr;
  OldEHSelectorSlot = CGF.EHSelectorSlot;
  CGF.EHSelectorSlot = nullptr;
  OldNormalCleanupDest = CGF.NormalCleanupDest;
  CGF.NormalCleanupDest = Address::invalid();

  CGF.pushFullExprCleanup<EndUnassocTaskFrame>(
      static_cast<CleanupKind>(NormalAndEHCleanup | LifetimeMarker | TaskExit),
      this);
}

CodeGenFunction::TaskFrameScope::~TaskFrameScope() {
  if (!CGF.CurSyncRegion)
    return;

  // Pop the taskframe.
  CGF.PopCleanupBlock();

  // Restore the alloca insertion point.
  {
    llvm::Instruction *Ptr = CGF.AllocaInsertPt;
    CGF.AllocaInsertPt = OldAllocaInsertPt;
    Ptr->eraseFromParent();
  }

  // Restore the original EH state.
  llvm::BasicBlock *NestedEHResumeBlock = CGF.EHResumeBlock;
  CGF.EHResumeBlock = OldEHResumeBlock;
  CGF.ExceptionSlot = OldExceptionSlot;
  CGF.EHSelectorSlot = OldEHSelectorSlot;
  CGF.NormalCleanupDest = OldNormalCleanupDest;

  if (TempInvokeDest) {
    if (llvm::BasicBlock *InvokeDest = CGF.getInvokeDest()) {
      TempInvokeDest->replaceAllUsesWith(InvokeDest);
    } else
      EmitTrivialLandingPad(CGF, TempInvokeDest);

    if (TempInvokeDest->use_empty())
      delete TempInvokeDest;
  }

  // If invocations in the parallel task led to the creation of EHResumeBlock,
  // we need to create for outside the task.  In particular, the new
  // EHResumeBlock must use an ExceptionSlot and EHSelectorSlot allocated
  // outside of the task.
  if (NestedEHResumeBlock) {
    if (!NestedEHResumeBlock->use_empty()) {
      // Translate the nested EHResumeBlock into an appropriate EHResumeBlock in
      // the outer scope.
      NestedEHResumeBlock->replaceAllUsesWith(
          CGF.getEHResumeBlock(
              isa<llvm::ResumeInst>(NestedEHResumeBlock->getTerminator())));
    }
    delete NestedEHResumeBlock;
  }
}

llvm::Instruction *CodeGenFunction::EmitSyncRegionStart() {
  // Start the sync region.  To ensure the syncregion.start call dominates all
  // uses of the generated token, we insert this call at the alloca insertion
  // point.
  llvm::Instruction *SRStart = llvm::CallInst::Create(
      CGM.getIntrinsic(llvm::Intrinsic::syncregion_start),
      "syncreg", AllocaInsertPt);
  return SRStart;
}
