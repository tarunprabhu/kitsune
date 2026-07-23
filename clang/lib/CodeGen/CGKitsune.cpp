//===- CGKitsune.cpp - Codegen for Kitsune's constructs -------------------===//
//
// TODO: Need to update LANL/Triad Copyright notice...
//
// Copyright (c) 2017, Los Alamos National Security, LLC.
// All rights reserved.
//
//  Copyright 2010. Los Alamos National Security, LLC. This software was
//  produced under U.S. Government contract DE-AC52-06NA25396 for Los
//  Alamos National Laboratory (LANL), which is operated by Los Alamos
//  National Security, LLC for the U.S. Department of Energy. The
//  U.S. Government has rights to use, reproduce, and distribute this
//  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
//  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
//  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
//  derivative works, such modified software should be clearly marked,
//  so as not to confuse it with the version available from LANL.
//
//  Additionally, redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the
//  following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of conditions and the following
//      disclaimer in the documentation and/or other materials provided
//      with the distribution.
//
//    * Neither the name of Los Alamos National Security, LLC, Los
//      Alamos National Laboratory, LANL, the U.S. Government, nor the
//      names of its contributors may be used to endorse or promote
//      products derived from this software without specific prior
//      written permission.
//
//  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
//  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
//  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
//
// "Codegen" (i.e. LLVM IR generation) for Kitsune's constructs
//
//===----------------------------------------------------------------------===//

#include "CGKitsune.h"
#include "CodeGenFunction.h"
#include "kitsune/Clang/ASTUtils.h"
#include "kitsune/Core/KitOptions.h"
#include "kitsune/Core/TTUtils.h"
#include "clang/AST/StmtKitsune.h"
#include "clang/Frontend/FrontendDiagnostic.h"

using namespace clang;
using namespace CodeGen;

/// Get the value of the tapir::target attribute if it was set. If the attribute
/// was not set, return the primary tapir target \p primaryTT. This will
/// will typically be the value of the --tapir command-line option.
llvm::TTID clang::CodeGen::getTTID(llvm::ArrayRef<const Attr *> attrs,
                                   llvm::TTID primaryTT) {
  // The TTAttr attribute is guaranteed to appear at most once, so it is safe
  // to return immediately when it is encountered.
  for (const Attr *attr : attrs) {
    if (const auto *ttAttr = dyn_cast<TTAttr>(attr)) {
      switch (ttAttr->getTT()) {
      case TTAttr::Nolo:
        return llvm::TTID::Nolo;
      case TTAttr::Cuda:
        return llvm::TTID::Cuda;
      case TTAttr::Hip:
        return llvm::TTID::Hip;
      case TTAttr::OpenCilk:
        return llvm::TTID::OpenCilk;
      case TTAttr::OpenMP:
        return llvm::TTID::OpenMP;
      case TTAttr::Pthreads:
        return llvm::TTID::Pthreads;
      case TTAttr::Qthreads:
        return llvm::TTID::Qthreads;
      case TTAttr::Serial:
        return llvm::TTID::Serial;
      case TTAttr::Custom:
        llvm_unreachable("Value of tapir target attribute cannot be 'custom'");
      }
      llvm_unreachable("getTTID: TTAttr not handled");
    }
  }
  return primaryTT;
}

/// Get the value of the kitsune::launch attribute if it was set. If the
/// attribute was not set, return 0.
unsigned clang::CodeGen::getLaunchTPB(llvm::ArrayRef<const Attr *> attrs) {
  // The KitsuneLaunch attribute is guaranteed to appear at most once, so it is
  // safe to return immediately when it is encountered.
  for (const Attr *attr : attrs)
    if (const auto *launchAttr = dyn_cast<KitsuneLaunchAttr>(attr))
      return launchAttr->getThreadsPerBlock();
  return 0;
}

llvm::Instruction *CodeGenFunction::EmitLabeledSyncRegionStart(StringRef SV) {
  // Start the sync region.  To ensure the syncregion.start call dominates all
  // uses of the generated token, we insert this call at the alloca insertion
  // point.
  llvm::Function *Func = CGM.getIntrinsic(llvm::Intrinsic::syncregion_start);
  llvm::Instruction *SRStart = llvm::CallInst::Create(
      Func->getFunctionType(), Func, SV, AllocaInsertPt->getIterator());
  return SRStart;
}

/// EmitSyncStmt - Emit a sync node.
void CodeGenFunction::EmitSyncStmt(const SyncStmt &S) {
  llvm::BasicBlock *ContinueBlock = createBasicBlock("sync.continue");

  // If this code is reachable then emit a stop point (if generating debug
  // info). We have to do this ourselves because we are on the "simple"
  // statement path.
  if (HaveInsertPoint())
    EmitStopPoint(&S);

  Builder.CreateSync(
      ContinueBlock,
      getOrCreateLabeledSyncRegion(S.getSyncVar())->getSyncRegionStart());
  EmitBlock(ContinueBlock);
}

void CodeGenFunction::EmitSpawnStmt(const SpawnStmt &S) {
  // Set up to perform a detach.
  SyncRegion *SR = getOrCreateLabeledSyncRegion(S.getSyncVar());

  llvm::BasicBlock *DetachedBlock = createBasicBlock("det.achd");
  llvm::BasicBlock *ContinueBlock = createBasicBlock("det.cont");

  llvm::AssertingVH<llvm::Instruction> OldAllocaInsertPt = AllocaInsertPt;
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "", DetachedBlock);

  Builder.CreateDetach(DetachedBlock, ContinueBlock, SR->getSyncRegionStart());

  EmitBlock(DetachedBlock);
  EmitStmt(S.getSpawnedStmt());

  Builder.CreateReattach(ContinueBlock, SR->getSyncRegionStart());

  llvm::Instruction *ptr = AllocaInsertPt;
  AllocaInsertPt = OldAllocaInsertPt;
  ptr->eraseFromParent();

  EmitBlock(ContinueBlock);
}

void CodeGenFunction::SetAllocaInsertPoint(llvm::BasicBlock *BB) {
  llvm::LLVMContext &Ctx = BB->getContext();
  llvm::Type *Int32Ty = llvm::Type::getInt32Ty(Ctx);
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);

  // This is a really hacky way of setting up an insertion point for
  // instructions outside of the builder used in CodeGenFunction. It works by
  // creating a dummy instructions in the given basic block, then using said
  // instruction to get an iterator at which to insert subsequent instructions.
  // This approach is used elsewhere in clang - which really ought not to be an
  // excuse for doing something this distasteful. Given how stateful clang is,
  // this is probably not even as bad as it gets.
  AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "", BB);
}

void CodeGenFunction::RestoreAllocaInsertPoint(llvm::Instruction *Saved) {
  // AllocaInsertPt will have been set to a dummy instruction whose sole
  // purpose was to act as an insertion point for other, actually relevant,
  // instructions. Before we reset the value, get rid of the dummy instruction.
  AllocaInsertPt->removeFromParent();
  AllocaInsertPt = Saved;
}

void CodeGenFunction::EmitIVLoad(const VarDecl *LoopVar,
                                 DeclMapByValueTy &IVDeclMap) {
  // The address corresponding to the IV
  Address IVAddress = LocalDeclMap.find(LoopVar)->second;
  LocalDeclMap.erase(LoopVar);

  QualType type = LoopVar->getType();
  llvm::SmallVector<llvm::Value *, 4> ValueVec;

  // Emit all the shallow copy loads and update
  switch (getEvaluationKind(type)) {
  case TEK_Scalar: {
    LValue IVLV = MakeAddrLValue(IVAddress, type);
    RValue IVRV = EmitLoadOfLValue(IVLV, LoopVar->getBeginLoc());
    ValueVec.push_back(IVRV.getScalarVal());
    break;
  }
  case TEK_Complex: {
    ComplexPairTy Val = EmitLoadOfComplex(MakeAddrLValue(IVAddress, type),
                                          LoopVar->getBeginLoc());
    ValueVec.push_back(Val.first);
    ValueVec.push_back(Val.second);
    break;
  }
  case TEK_Aggregate: {
    if (auto *STy = dyn_cast<llvm::StructType>(IVAddress.getElementType())) {
      for (unsigned I = 0, E = STy->getNumElements(); I != E; ++I) {
        Address EltPtr = Builder.CreateStructGEP(IVAddress, I);
        llvm::Value *Elt = Builder.CreateLoad(EltPtr);
        ValueVec.push_back(Elt);
      }
    } else {
      LValue IVLV = MakeAddrLValue(IVAddress, type);
      RValue IVRV = EmitLoadOfLValue(IVLV, LoopVar->getBeginLoc());
      ValueVec.push_back(IVRV.getScalarVal());
    }
    break;
  }
  }

  // Capture the mapping from LoopVar to the old address and new vector of
  // Value*'s.
  IVDeclMap.insert({LoopVar, {IVAddress, ValueVec}});
}

void CodeGenFunction::EmitThreadSafeIV(
    const VarDecl *IV, const llvm::SmallVectorImpl<llvm::Value *> &Values) {
  AutoVarEmission LVEmission = EmitAutoVarAlloca(*IV);
  EmitAutoVarCleanups(LVEmission);

  QualType type = IV->getType();
  Address Loc = LVEmission.getObjectAddress(*this);
  LValue LV = MakeAddrLValue(Loc, type);

  // Make sure the LValue isn't garbage collected
  LV.setNonGC(true);

  switch (getEvaluationKind(type)) {
  case TEK_Scalar:
    EmitStoreOfScalar(Values[0], LV, true);
    break;
  case TEK_Complex:
    EmitStoreOfComplex({Values[0], Values[1]}, LV, true);
    break;
  case TEK_Aggregate:
    if (auto *STy = dyn_cast<llvm::StructType>(Loc.getElementType())) {
      for (unsigned I = 0, E = STy->getNumElements(); I != E; ++I) {
        Address EltPtr = Builder.CreateStructGEP(Loc, I);
        llvm::Value *Elt = Values[I];
        Builder.CreateStore(Elt, EltPtr);
      }
    } else {
      EmitStoreOfScalar(Values[0], LV, /*isInit*/ true);
    }
    break;
  }
}

void CodeGenFunction::RestoreDeclMap(const VarDecl *IV, const Address IVAddr) {
  LocalDeclMap.erase(IV);
  LocalDeclMap.insert({IV, IVAddr});
}

void CodeGenFunction::EmitForallStmt(const ForallStmt &S,
                                     ArrayRef<const Attr *> Attrs) {
  const llvm::driver::KitOptions &KitOpts = CGM.getKitOpts();
  assert(KitOpts.getTTID().has_value() && "TTID not set in Kitsune options");

  llvm::TTID TT = getTTID(Attrs, *KitOpts.getTTID());

  // The tapir target *must* be set before any other attributes are set in
  // LoopStack.
  if (TT != llvm::TTID::Nolo) {
    LoopStack.setTapirTarget(TT);
    LoopStack.setTapirSpawnStrategy(getSpawnStrategyFor(TT));
    LoopStack.setTapirLoopName(getNameFor(S, Attrs, getContext()));
  }

  if (isGPUTT(TT))
    if (unsigned TPB = getLaunchTPB(Attrs))
      LoopStack.setLoopThreadsPerBlock(TPB);

  // New basic blocks and jump destinations with Tapir terminators
  llvm::BasicBlock *Detach = createBasicBlock("forall.detach");
  JumpDest Reattach = getJumpDestInCurrentScope("forall.reattach");
  JumpDest Sync = getJumpDestInCurrentScope("forall.sync");

  // Declarations for capturing the IV vardecl to old and new llvm Values as
  // well as the alloca insertion point which we need to change and change back
  DeclMapByValueTy IVDeclMap; // map from Vardecl to {IV, thread safe IV vector}
  llvm::AssertingVH<llvm::Instruction> OldAllocaInsertPt = AllocaInsertPt;

  // emit the sync region
  PushSyncRegion();
  llvm::Instruction *SRStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SRStart);
  // See if we have any launch attributes to handle before we start loop body.

  JumpDest LoopExit = getJumpDestInCurrentScope("forall.end");
  LexicalScope ForScope(*this, S.getSourceRange());

  // Evaluate the initialization before the loop.
  EmitStmt(S.getInit());

  // In a parallel loop there will always be a condition block so there is no
  // no need to test.
  JumpDest Condition = getJumpDestInCurrentScope("forall.cond");
  llvm::BasicBlock *CondBlock = Condition.getBlock();
  EmitBlock(CondBlock);

  const SourceRange &R = S.getSourceRange();
  LoopStack.push(CondBlock, CGM.getContext(), CGM.getCodeGenOpts(), Attrs,
                 SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  // In a parallel loop, there will always be an increment block.
  JumpDest Increment = getJumpDestInCurrentScope("forall.inc");

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(LoopExit, Reattach));

  // Create a cleanup scope for the condition variable cleanups.
  // We don't need this unless we allow condition scope variables
  LexicalScope ConditionScope(*this, S.getSourceRange());

  // If the for statement has a condition scope, emit the local variable
  // declaration.
  // Presently, we don't support condition variables, but we should :-)
  if (S.getConditionVariable())
    EmitDecl(*S.getConditionVariable());

  llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
  // If there are any cleanups between here and the loop-exit scope,
  // create a block to stage a loop exit along.
  if (ForScope.requiresCleanups())
    ExitBlock = createBasicBlock("forall.cond.cleanup");

  // As long as the condition is true, iterate the loop.
  llvm::BasicBlock *ForBody = createBasicBlock("forall.body");

  // C99 6.8.5p2/p4: The first substatement is executed if the expression
  // compares unequal to 0.  The condition must be a scalar type.
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
  Builder.CreateCondBr(
      BoolCondVal, Detach, Sync.getBlock(),
      createProfileWeightsForLoop(S.getCond(), getProfileCount(S.getBody())));

  if (ExitBlock != LoopExit.getBlock()) {
    EmitBlock(ExitBlock);
    EmitBranchThroughCleanup(Sync);
  }

  // Emits the detach block for parallel execution along with its Tapir
  // terminator. This is where we capture the induction variable by value and
  // store it on the stack of the calling thread.
  EmitBlock(Detach);

  // Extract the DeclStmt from the statement init. This is guaranteed to exist.
  const DeclStmt *DS = cast<DeclStmt>(S.getInit());

  // Create threadsafe induction variables before the detach and put them in
  // IVDeclMap
  for (auto *DI : DS->decls())
    EmitIVLoad(dyn_cast<VarDecl>(DI), IVDeclMap);

  // create the detach terminator
  Builder.CreateDetach(ForBody, Increment.getBlock(), SRStart);

  EmitBlock(ForBody);
  incrementProfileCounter(&S);

  {
    // Create a separate cleanup scope for the body, in case it is not
    // a compound statement.
    RunCleanupsScope BodyScope(*this);

    // In this block of code, we change the alloca insert point so that the
    // alloca's happen after the detach and within the body block. This makes
    // sure each thread has its own local copy of the induction variable. We
    // also need to store the thread safe value from the calling thread into
    // this local copy. In EmitThreadSafeIV, we use AutoVarAlloca so any codegen
    // in the body automatically and correctly mapped to the local thread
    // safe copy of the induction variable.
    SetAllocaInsertPoint(ForBody);

    // Emit the thread safe induction variables and initialize them by value.
    for (const auto &ivp : IVDeclMap)
      EmitThreadSafeIV(ivp.first, ivp.second.second);

    EmitStmt(S.getBody());
  }

  // Unwind the codegen of the induction variable from the current local thread
  // safe copy back to the original induction variable. We also need to emit the
  // reattach block and reset the alloca insertion point.

  // Restore induction variable mappings after emitting body, and before the
  // increment
  for (const auto &ivp : IVDeclMap)
    RestoreDeclMap(ivp.first, ivp.second.first);

  // Emit the reattach block.
  EmitBlock(Reattach.getBlock());
  Builder.CreateReattach(Increment.getBlock(), SRStart);

  // Reset the alloca insertion point.
  RestoreAllocaInsertPoint(OldAllocaInsertPt);

  // Emit the increment.
  EmitBlock(Increment.getBlock());
  EmitStmt(S.getInc());

  BreakContinueStack.pop_back();

  ConditionScope.ForceCleanup();

  EmitStopPoint(&S);
  EmitBranch(CondBlock);

  ForScope.ForceCleanup();

  LoopStack.pop();

  // Emit the Sync block and terminator.
  EmitBlock(Sync.getBlock());
  Builder.CreateSync(LoopExit.getBlock(), SRStart);
  PopSyncRegion();

  // Emit the fall-through block.
  EmitBlock(LoopExit.getBlock(), true);
}

void CodeGenFunction::EmitCXXForallRangeStmt(const CXXForallRangeStmt &S,
                                             ArrayRef<const Attr *> Attrs) {
  const llvm::driver::KitOptions &KitOpts = CGM.getKitOpts();
  assert(KitOpts.getTTID().has_value() && "TTID not set in Kitsune options");

  llvm::TTID TT = getTTID(Attrs, *KitOpts.getTTID());

  // The tapir target *must* be set before any other attributes are set in
  // LoopStack.
  if (TT != llvm::TTID::Nolo) {
    LoopStack.setTapirTarget(TT);
    LoopStack.setTapirSpawnStrategy(getSpawnStrategyFor(TT));
    LoopStack.setTapirLoopName(getNameFor(S, Attrs, getContext()));
  }

  if (isGPUTT(TT))
    if (unsigned TPB = getLaunchTPB(Attrs))
      LoopStack.setLoopThreadsPerBlock(TPB);

  // New basic blocks and jump destinations with Tapir terminators.
  llvm::BasicBlock *Detach = createBasicBlock("forall.detach");
  JumpDest Reattach = getJumpDestInCurrentScope("forall.reattach");
  JumpDest LoopExit = getJumpDestInCurrentScope("forall.sync");

  // Declarations for capturing the IV vardecl to old and new llvm Values as
  // well as the alloca insertion point which we need to change and change back
  DeclMapByValueTy IVDeclMap; // map from Vardecl to {IV, thread safe IV}
  llvm::AssertingVH<llvm::Instruction> OldAllocaInsertPt = AllocaInsertPt;

  // Emit the sync region.
  PushSyncRegion();
  llvm::Instruction *SRStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SRStart);

  llvm::BasicBlock *End = createBasicBlock("forall.end");

  LexicalScope ForScope(*this, S.getSourceRange());

  // Evaluate the first pieces before the loop.
  if (S.getInit())
    EmitStmt(S.getInit());
  EmitStmt(S.getRangeStmt());
  EmitStmt(S.getBeginStmt());
  EmitStmt(S.getEndStmt());
  EmitStmt(S.getIndexStmt());
  EmitStmt(S.getIndexEndStmt());

  // In a parallel loop there will always be a condition block, so there is no
  // need to test
  llvm::BasicBlock *CondBlock = createBasicBlock("forall.cond");
  EmitBlock(CondBlock);

  const SourceRange &R = S.getSourceRange();
  LoopStack.push(CondBlock, CGM.getContext(), CGM.getCodeGenOpts(), Attrs,
                 SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  // If there are any cleanups between here and the loop-exit scope, create a
  // block to stage a loop exit along.
  llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
  if (ForScope.requiresCleanups())
    ExitBlock = createBasicBlock("forall.cond.cleanup");

  // The loop body, consisting of the specified body and the loop variable.
  llvm::BasicBlock *ForBody = createBasicBlock("forall.body");

  // The body is executed if the expression, contextually converted to bool, is
  // true.
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
  llvm::MDNode *Weights =
      createProfileWeightsForLoop(S.getCond(), getProfileCount(S.getBody()));
  Builder.CreateCondBr(BoolCondVal, Detach, ExitBlock, Weights);

  if (ExitBlock != LoopExit.getBlock()) {
    EmitBlock(ExitBlock);
    EmitBranchThroughCleanup(LoopExit);
  }

  // Emits the detach block for parallel execution along with its Tapir
  // terminator. This is where we capture the induction variable by value and
  // store it on the stack of the calling thread.

  // Emit the (currently empty) detach block.
  EmitBlock(Detach);

  // Extract the DeclStmt from the statement init.
  const DeclStmt *DS = cast<DeclStmt>(S.getIndexStmt());

  // Create threadsafe induction variables before the detach and put them in
  // IVDeclMap
  for (auto *DI : DS->decls())
    EmitIVLoad(dyn_cast<VarDecl>(DI), IVDeclMap);

  // Create a block for the increment. In case of a 'continue', we jump there.
  llvm::BasicBlock *Increment = createBasicBlock("forall.inc");

  // Create the detach terminator
  Builder.CreateDetach(ForBody, Increment, SRStart);

  EmitBlock(ForBody);
  incrementProfileCounter(&S);

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(LoopExit, Reattach));

  {
    // Create a separate cleanup scope for the loop variable and body.
    LexicalScope BodyScope(*this, S.getSourceRange());

    // Change the alloca insert point so that the alloca's happen after the
    // detach and within the body block. This makes sure each thread has its own
    // local copy of the induction variable. We also need to store the thread
    // safe value from the calling thread into this local copy. In
    // EmitThreadSafeIV, we use AutoVarAlloca so any codegen in the body
    // automatically and correctly mapped to the local thread safe copy of the
    // induction variable.
    SetAllocaInsertPoint(ForBody);

    // Emit the thread safe induction variables and initialize them by value.
    for (const auto &ivp : IVDeclMap)
      EmitThreadSafeIV(ivp.first, ivp.second.second);

    EmitStmt(S.getLoopVarStmt());
    EmitStmt(S.getBody());
  }

  // Unwind the codegen of the induction variable from the current local thread
  // safe copy back to the original induction variable. We also need to emit the
  // reattach block and reset the alloca insertion point.

  // Restore induction variable mappings after emitting body, and before the
  // increment.
  for (const auto &ivp : IVDeclMap)
    RestoreDeclMap(ivp.first, ivp.second.first);

  EmitBlock(Reattach.getBlock());
  Builder.CreateReattach(Increment, SRStart);

  // Reset the alloca insertion point.
  RestoreAllocaInsertPoint(OldAllocaInsertPt);

  EmitStopPoint(&S);

  EmitBlock(Increment);
  EmitStmt(S.getInc());

  BreakContinueStack.pop_back();
  EmitBranch(CondBlock);
  ForScope.ForceCleanup();
  LoopStack.pop();

  // Emit the Sync block and terminator.
  EmitBlock(LoopExit.getBlock());
  Builder.CreateSync(End, SRStart);
  PopSyncRegion();

  EmitBlock(End, true);
}
