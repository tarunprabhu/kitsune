/**
 ***************************************************************************
 * TODO: Need to update LANL/Triad Copyright notice...
 *
 * Copyright (c) 2017, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 *
 ***************************************************************************/
#include <cstdio>
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "CodeGenFunction.h"

using namespace clang;
using namespace CodeGen;

// DWS Worth discussing why we want an anonymous namespace and what belongs here
// as opposed to putting in CodeGenFunction.h
namespace {

  // Part of the challenge in unwinding all the Kokkos
  // C++ details is getting through a bunch of implicit
  // goop in the AST. Fortunately Clang can help here
  // a lot as it has mechanisms to toss out implicit
  // constructs and get us closer to the underlying
  // expression.  Pulled out of inline code to allow
  // for more agressive simplications/modifications
  // if needed (and reuse across different constructs).
  static const Expr *SimplifyExpr(const Expr *E) {
    return E->IgnoreImplicit()->IgnoreImpCasts();
  }

  // Break apart the various components of a Kokkos
  // parallel_for.  This boils down to tearing apart
  // a CallExpr.  It helps that at this point we've
  // already passed all the "goodness" in semantic
  // analysis and other frontend stages.  However,
  // there are aspects of what you can do generally
  // in Kokkos and what we can tear apart here (both
  // big picture and in terms of the state of the
  // current implementation).  A current set of
  // limitations are:
  //
  //   - Functors are not supported (primarily given
  //     challenges around separate compilation units)
  //   - 'Named constructs', which Kokkos uses for profiling,
  //     are extracted but unused.
  //   - There are likely many holes in types that are legal
  //     constructs but not yet captured/covered in the
  //     current set of transformations currently implemented.
  //     The code tries to avoid crashing/asserting and instead
  //     falls back to the standard C++ mechanisms. We will start
  //     ramp up testing as we get more exposure to use cases.
  //
  static bool
  ParseAndValidateParallelFor(const CallExpr* CE,
					   std::string &CN, const Expr *& BE,
					   const LambdaExpr *& LE,
             DiagnosticsEngine &Diags)
  {
    // Recognized constructs:
    //
    //   1. parallel_for(N, lambda_expr...);
    //
    //   2. parallel_for("name", N, lambda_expr...);

    unsigned int curArgIndex = 0;

    // Check for a 'named' construct.  NOTE: we assume
    // this only comes in the form of a string literal.
    const Expr *OE = CE->getArg(curArgIndex); // Original expression
    const Expr *SE = SimplifyExpr(OE);        // Simplified expression.

    if (SE->getStmtClass() == Expr::CXXConstructExprClass) {
      const CXXConstructExpr *CXXCE = dyn_cast<CXXConstructExpr>(SE);
      SE = CXXCE->getArg(0)->IgnoreImplicit();
      if (SE->getStmtClass() == Expr::StringLiteralClass) {
        CN = dyn_cast<StringLiteral>(SE)->getString().str();
        curArgIndex++;
        OE = CE->getArg(curArgIndex);
        SE = SimplifyExpr(OE);
      }
    }

    // Parse the bounds portion of the parallel_for.
    // Note: This can take several forms and it is likely we've
    // missed some cases...
    if (SE->getStmtClass() == Expr::IntegerLiteralClass) {
      BE = OE;
      curArgIndex++;
      OE = CE->getArg(curArgIndex);
      SE = SimplifyExpr(OE);
    } else if (SE->getStmtClass() == Expr::BinaryOperatorClass) {
      BE = OE;
      curArgIndex++;
      OE = CE->getArg(curArgIndex);
      SE = SimplifyExpr(OE);
    } else if (SE->getStmtClass() == Expr::DeclRefExprClass) {
      BE = OE;
      curArgIndex++;
      OE = CE->getArg(curArgIndex);
      SE = SimplifyExpr(OE);
    } else if (SE->getStmtClass() == Expr::CallExprClass) {
      BE = OE;
      curArgIndex++;
      OE = CE->getArg(curArgIndex);
      SE = SimplifyExpr(OE);
    } else if (SE->getStmtClass() == Expr::CXXTemporaryObjectExprClass) {
      BE = SE;
      curArgIndex++;
      OE = CE->getArg(curArgIndex);
      SE = SimplifyExpr(OE);
    } else {
      Diags.Report(SE->getExprLoc(), diag::warn_kokkos_unknown_stmt_class);
      BE = nullptr;
    }

    // Parse the lambda expression in the parallel_for
    LE = dyn_cast<LambdaExpr>(SE);

    // Perform semantic analysis of the derived constructs

    // Check the bounds expression
    if (BE == nullptr) {
      // We didn't get a known bounds expression back -- this is most likely
      // due to some type of expression that we have yet to deal with.  We're
      // going to have to uncover these on a case-by-cases basis.
      Diags.Report(CE->getExprLoc(), diag::warn_kokkos_unknown_bounds_expr);
      return false;
    }

    // Check the lambda expression is at least well formed
    if (LE == nullptr) {
      // The parallel_for doesn't have a (recognizable) lambda expression.
      Diags.Report(CE->getExprLoc(), diag::warn_kokkos_no_functor);
      return false;
    }

    // Maker sure the lambda has defined arguments
    if (LE->getCallOperator()->getNumParams()==0){
      // The lambda has no arguments, and gets all variables by capture
      Diags.Report(CE->getExprLoc(), diag::warn_kokkos_no_functor_args);
      return false;
     
    }
    // Everything was parsed correctly
    return true;
  }


}

// Emit the ParmVarDecl defined in a Kokkos lambda and initialize
// the resulting values
void CodeGenFunction::EmitAndInitializeIV(const ParmVarDecl* IV){

  // Just emit the induction variable from the ParmVarDecl
  EmitVarDecl(*IV);
  
  // Define zero (at some point we will want to generalize this)
  llvm::Value *Zero = llvm::ConstantInt::get(ConvertType(IV->getType()), 0);

  // Store zero into the address
  Builder.CreateStore(Zero, GetAddrOfLocalVar(IV));
}

// Sort through what sort of Kokkos construct we're looking at
// and work on transforming it into a Tapir-centric lowering.
//
bool CodeGenFunction::EmitKokkosConstruct(const CallExpr *CE,
                ArrayRef<const Attr *> Attrs) {
  assert(CE != 0 && "CodeGenFunction::EmitKokkosConstruct: null callexpr passed!");

  const FunctionDecl *Func = CE->getDirectCallee();
  assert(Func != 0 && "Kokkos construct doesn't have a function declaration!");

  if (Func->getQualifiedNameAsString() == "Kokkos::parallel_for") {
    return EmitKokkosParallelFor(CE, Attrs);
  } else if (Func->getQualifiedNameAsString() == "Kokkos::parallel_reduce") {
    return EmitKokkosParallelReduce(CE, Attrs);
  } else {
    return false;
  }
}  // hidden/local namespace

/*
std::vector<const ParmVarDecl*>
CodeGenFunction::EmitKokkosParallelForInductionVar(const LambdaExpr *Lambda) {
  const CXXMethodDecl *MD = Lambda->getCallOperator();
  assert(MD && "EmitKokkosParallelFor() -- bad method decl from labmda call.");
  
  std::vector<const ParmVarDecl*> params;
  
  for (unsigned int i = 0; i<MD->getNumParams(); i++) {
    const ParmVarDecl *InductionVarDecl = MD->getParamDecl(i);
    assert(InductionVarDecl && "EmitKokkosParallelFor() -- bad loop variable decl!");
    
    //EmitVarDecl(*InductionVarDecl);
    params.push_back(InductionVarDecl);
  }
  
  return params;
}
*/

llvm::Value* CodeGenFunction::EmitKokkosParallelForCond(const ParmVarDecl *IV, 
  const Expr *BoundsExpr) {

  llvm::Value *LoopEnd = nullptr;
  if (BoundsExpr->getStmtClass() == Expr::BinaryOperatorClass) {
    RValue RV = EmitAnyExpr(BoundsExpr);
    LoopEnd = RV.getScalarVal();
  } else { 
    LoopEnd = EmitScalarExpr(BoundsExpr);
  }

  llvm::Type  *InductionVarTy = ConvertType(IV->getType());
  unsigned NBits  = LoopEnd->getType()->getPrimitiveSizeInBits();
  unsigned LVBits = InductionVarTy->getPrimitiveSizeInBits();

  // We may need to truncate/extend the range to get it to match
  // the type of loop variable.
  if (NBits > LVBits) {
    LoopEnd = Builder.CreateTrunc(LoopEnd, InductionVarTy);
  } else if (NBits < LVBits) {
    LoopEnd = Builder.CreateZExt(LoopEnd, InductionVarTy);
  } else {
    // bit count matches, nothing to do...
  }

  llvm::Value *InductionVal = Builder.CreateLoad(GetAddrOfLocalVar(IV));
  return Builder.CreateICmpULT(InductionVal, LoopEnd);
}

void CodeGenFunction::EmitKokkosIncrement(const ParmVarDecl *IV){
  // load the induction variable
  llvm::Value *IncVal = Builder.CreateLoad(GetAddrOfLocalVar(IV));

  // create the number "1" with the correct type
  llvm::Value *One = llvm::ConstantInt::get(ConvertType(IV->getType()), 1);

  // emit the add one
  IncVal = Builder.CreateAdd(IncVal, One);

  // emit the store back into the induction variable
  Builder.CreateStore(IncVal, GetAddrOfLocalVar(IV));
}

// DWS not sure if this shouldn't be a void return type
bool CodeGenFunction::EmitKokkosParallelFor(const CallExpr *CE,
              ArrayRef<const Attr *> KokkosAttrs) {
  /////////////////////////////////////////////////////////////////////////////
  // Code Modifications necessary for implementing parallel loops not required
  // by serial loops.

  // New basic blocks and jump destinations with Tapir terminators
  llvm::BasicBlock* Detach = createBasicBlock("kokkos.detach");
  JumpDest Reattach = getJumpDestInCurrentScope("kokkos.reattach");
  JumpDest Sync = getJumpDestInCurrentScope("kokkos.sync");
  
  // Declarations for capturing the IV vardecl to old and new llvm Values as
  // well as the alloca insertion point which we need to change and change back
  DeclMapByValueTy IVDeclMap; // map from Vardecl to {IV, thread safe IV vector}
  llvm::AssertingVH<llvm::Instruction> OldAllocaInsertPt = AllocaInsertPt;
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);

  // emit the sync region
  PushSyncRegion();
  llvm::Instruction *SRStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SRStart);
  LoopStack.setSpawnStrategy(LoopAttributes::DAC);

  // Parse and validate the parallel for
  std::string      PFName; // construct name (for kokkos profiling)
  const Expr       *BE = nullptr; // "bounds" expression
  const LambdaExpr *Lambda = nullptr; // the lambda
  DiagnosticsEngine &Diags = CGM.getDiags();
  if (!ParseAndValidateParallelFor(CE, PFName, BE, Lambda, Diags)) return true;
  ArrayRef<ParmVarDecl*> DS{Lambda->getCallOperator()->parameters()};

  // If we find an MDRangePolicy we need to codegen a nested loop instead
  if (const CXXTemporaryObjectExpr *CXXTO = dyn_cast<CXXTemporaryObjectExpr>(BE))
    if (CXXTO->getBestDynamicClassType()->getNameAsString() == "MDRangePolicy") 
      return EmitKokkosParallelForMD(CE, PFName, BE, Lambda, KokkosAttrs);    

  // End of parallel modification code block
  /////////////////////////////////////////////////////////////////////////////

  JumpDest LoopExit = getJumpDestInCurrentScope("kokkos.end");

  LexicalScope ForScope(*this, CE->getSourceRange());

  // Evaluate the initialization before the loop. This is the analog of
  // EmitStmt(S.getInit());
  // The first step is to extract the argument to the lambda and transform it into 
  // the loop induction variable.  As part of this we assume the following are true
  // about the parallel_for:
  //    1. The iterator can be assigned a value of zero. 
  //    2. We ignore the details of what is captured by the lambda.
  // 
  // TODO: Do we need to "relax" these assumptions to support broader code coverage?
  // This is 'equivalent' to the Init statement in a traditional for loop (e.g. int i = 0). 
  for (const ParmVarDecl *IV : DS)
   EmitAndInitializeIV(IV);

  // In a parallel loop there will always be a condition block
  // so there is no need to test
  JumpDest Condition = getJumpDestInCurrentScope("kokkos.cond"); 
  llvm::BasicBlock *CondBlock = Condition.getBlock();
  EmitBlock(CondBlock);

  const SourceRange &R = CE->getSourceRange();
  LoopStack.push(CondBlock, CGM.getContext(), CGM.getCodeGenOpts(), KokkosAttrs,
                 SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  // In a parallel loop, there will always be an increment block
  JumpDest Increment = getJumpDestInCurrentScope("kokkos.inc");

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(LoopExit, Reattach));

  // Create a cleanup scope for the condition variable cleanups.
  // We don't need this unless we allow condition scope variables
  LexicalScope ConditionScope(*this, CE->getSourceRange());
  
  llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
  // If there are any cleanups between here and the loop-exit scope,
  // create a block to stage a loop exit along.
  if (ForScope.requiresCleanups())
    ExitBlock = createBasicBlock("kokkos.cond.cleanup");

  // As long as the condition is true, iterate the loop.
  llvm::BasicBlock *ForBody = createBasicBlock("kokkos.body");

  // C99 6.8.5p2/p4: The first substatement is executed if the expression
  // compares unequal to 0.  The condition must be a scalar type.
  // Create the conditional.
  // DWS fix this to handle more than one induction variable
  // right now, this buggy codegen will emit a branch per variable
  // which is probably correct for nested loops, but I don't think
  // anything else will work...
  for (const ParmVarDecl *IV : DS){
    llvm::Value *BoolCondVal = EmitKokkosParallelForCond(IV, BE);
    Builder.CreateCondBr(
        BoolCondVal, Detach, Sync.getBlock());
    // DWS fix profile weights
    // ,createProfileWeightsForLoop(S.getCond(), getProfileCount(S.getBody()))
  }

  if (ExitBlock != LoopExit.getBlock()) {
    EmitBlock(ExitBlock);
    EmitBranchThroughCleanup(Sync);
  }

  /////////////////////////////////////////////////////////////////////////////
  // The following block of code emits the detach block for parallel execution
  // along with its Tapir terminator. This is where we capture the induction 
  // variable by value and store it on the stack of the calling thread.

  EmitBlock(Detach);

  // Create threadsafe induction variables before the detach and put them in IVDeclMap
  for (const ParmVarDecl *DI : DS) 
    EmitIVLoad(dyn_cast<VarDecl>(DI), IVDeclMap);

  // create the detach terminator
  Builder.CreateDetach(ForBody, Increment.getBlock(), SRStart);
  
  // End of parallel modification code block
  /////////////////////////////////////////////////////////////////////////////

  EmitBlock(ForBody);

  incrementProfileCounter(CE);

  {
    // DWS need to explain this, because right now I have no idea, but 
    // is absolutely necessary not to segfault during codegen :-)
    InKokkosConstruct=true;

    // Create a separate cleanup scope for the body, in case it is not
    // a compound statement.
    RunCleanupsScope BodyScope(*this);

    ///////////////////////////////////////////////////////////////////////////
    // In this block of code, we change the alloca insert point so that the
    // alloca's happen after the detach and within the body block. This makes
    // sure each thread has its own local copy of the induction variable. We
    // also need to store the thread safe value from the calling thread into
    // this local copy. In EmitThreadSafeIV, we use AutoVarAlloca so any codegen
    // in the body automatically and correctly mapped to the local thread
    // safe copy of the induction variable.

    // change the alloca insert point to the body block
    SetAllocaInsertPoint(Undef, ForBody);

    // emit the thread safe induction variables and initialize them by value
    for (const auto &ivp : IVDeclMap) 
      EmitThreadSafeIV(ivp.first, ivp.second.second);

    // End of parallel modification code block
    ///////////////////////////////////////////////////////////////////////////

    // emit the body of the lambda expression
    EmitStmt(Lambda->getBody());

    InKokkosConstruct=false;
  }

  /////////////////////////////////////////////////////////////////////////////
  // In this block of code, we need to unwind the codegen of the induction
  // variable from the current local thread safe copy back to the original
  // induction variable. We also need to emit the reattach block and reset the
  // alloca insertion point.

  // Restore induction variable mappings after emitting body, and before
  // the increment
  for (const auto &ivp : IVDeclMap) 
    RestoreDeclMap(ivp.first, ivp.second.first);

  // emit the reattach block
  EmitBlock(Reattach.getBlock());
  Builder.CreateReattach(Increment.getBlock(), SRStart);

  // reset the alloca insertion point
  AllocaInsertPt->removeFromParent();
  AllocaInsertPt = OldAllocaInsertPt; 

  // End of parallel modification code block
  /////////////////////////////////////////////////////////////////////////////

  // Emit the increment.
  EmitBlock(Increment.getBlock());

  // Emit the increments, which is the Kokkok analog of 
  // EmitStmt(S.getInc());
  for (const ParmVarDecl *IV : DS)
      EmitKokkosIncrement(IV);

  BreakContinueStack.pop_back();

  ConditionScope.ForceCleanup();

  EmitStopPoint(CE);
  EmitBranch(CondBlock);

  ForScope.ForceCleanup();

  LoopStack.pop();

  // Emit the Sync block and terminator
  EmitBlock(Sync.getBlock());
  Builder.CreateSync(LoopExit.getBlock(), SRStart);

  // Emit the fall-through block.
  EmitBlock(LoopExit.getBlock(), true);

  // DWS remove after type change???
  return true;
}

// The builds an MDRange parallel_for (basically, a loop with multiple dimensions).
// The function is rather simple. We basically break down the MDRange object, and then call a recursive function
// that does the fun work of creating a loop
//
bool CodeGenFunction::EmitKokkosParallelForMD(const CallExpr *CE, std::string PFName, const Expr *BE, const LambdaExpr *Lambda,
            ArrayRef<const Attr *> KokkosAttrs) {
    
  // TODO: Need to add code to process any attributes (KokkosAttrs).
  
  // Build the queue of dimensions
  std::vector<const Expr *> DimQueue;
  std::vector<const Expr *> StartQueue;
  
  const CXXTemporaryObjectExpr *CXXTO = dyn_cast<CXXTemporaryObjectExpr>(BE);
  const InitListExpr *StartingBounds = dyn_cast<InitListExpr>(CXXTO->getArg(0)->IgnoreImplicit());
  const InitListExpr *UpperBounds = dyn_cast<InitListExpr>(CXXTO->getArg(1)->IgnoreImplicit());
  
  // The starting and ending bounds should be the same length
  std::vector<std::pair<const Expr*, const Expr*>> BoundsList;
  
  if (StartingBounds->getNumInits() != UpperBounds->getNumInits()) {
    return false;
  }
  
  for (unsigned int i = 0; i<StartingBounds->getNumInits(); i++) {
    const Expr *start = StartingBounds->getInit(i)->IgnoreImplicit();
    const Expr *end = UpperBounds->getInit(i)->IgnoreImplicit();
    std::pair<const Expr*, const Expr*> pair(start, end);
    BoundsList.push_back(pair);
  }
  
  // Get the induction variables
  // DWS fix
  std::vector<const ParmVarDecl*> params;// = EmitKokkosParallelForInductionVar(Lambda);
  
  // Build the inner loops, and eventually the body
  std::vector<std::pair<llvm::Value*, llvm::AllocaInst*>> TLIVarList;
  return EmitKokkosInnerLoop(CE, Lambda, nullptr, BoundsList, params, TLIVarList, KokkosAttrs);
}

// This is in charge of building an inner loop. It works as a recursive function to allow the loops
// to actually end up being nested
//
// This should be usuable by any function that requires inner loops
//
bool CodeGenFunction::EmitKokkosInnerLoop(const CallExpr *CE, const LambdaExpr *Lambda,
            llvm::BasicBlock *TopBlock,
            std::vector<std::pair<const Expr*, const Expr*>> BoundsList,
            std::vector<const ParmVarDecl*> params,
            std::vector<std::pair<llvm::Value*, llvm::AllocaInst*>> TLIVarList,
            ArrayRef<const Attr *> KokkosAttrs) {
  // Load the data we need
  int pos = BoundsList.size();
  const Expr *BE = BoundsList.front().second;
  const Expr *SE = BoundsList.front().first;
  BoundsList.erase(BoundsList.begin());

  const ParmVarDecl *InductionVarDecl = params.front();
  params.erase(params.begin());
  
  llvm::BasicBlock *InductionSet = createBasicBlock("kokkos.forall.set" + std::to_string(pos));
  JumpDest Condition = getJumpDestInCurrentScope("kokkos.forall.cond" + std::to_string(pos));
  llvm::BasicBlock *Detach = createBasicBlock("kokkos.forall.detach" + std::to_string(pos));
  llvm::BasicBlock *PForBody = createBasicBlock("kokkos.forall.body" + std::to_string(pos));
  JumpDest Reattach = getJumpDestInCurrentScope("kokkos.forall.reattach" + std::to_string(pos));
  llvm::BasicBlock *Increment = createBasicBlock("kokkos.forall.inc" + std::to_string(pos));
  JumpDest Cleanup = getJumpDestInCurrentScope("kokkos.forall.cond.cleanup" + std::to_string(pos));
  JumpDest Sync = getJumpDestInCurrentScope("kokkos.forall.sync" + std::to_string(pos));
  llvm::BasicBlock *End = createBasicBlock("kokkos.forall.end" + std::to_string(pos));
  
  // Extract a conveince block and setup the lexical scope based on 
  // the lambda's source range. 
  llvm::BasicBlock *ConditionBlock = Condition.getBlock();

  const SourceRange &R = CE->getSourceRange();
  LexicalScope PForScope(*this, R);

  // Now we can start the dirty work of transforming the lambda into a
  // for loop.


  // The first step is to extract the argument to the lambda and transform it into
  // the loop induction variable.  As part of this we assume the following are true
  // about the parallel_for:
  //    1. The iterator can be assigned a value of zero.
  //    2. We ignore the details of what is captured by the lambda.
  //
  // TODO: Do we need to "relax" these assumptions to support broader code coverage?
  // This is 'equivalent' to the Init statement in a traditional for loop (e.g. int i = 0). 

   // Create the sync region.
  PushSyncRegion();
  llvm::Instruction *SRStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SRStart);
  
  // Set the induction variable's starting point
  EmitBlock(InductionSet);
  EmitVarDecl(*InductionVarDecl);
  llvm::Value *LoopStart = EmitScalarExpr(SE);
  Builder.CreateStore(LoopStart, GetAddrOfLocalVar(InductionVarDecl));

  // TODO: Need to check attributes for spawning strategy.
  LoopStack.setSpawnStrategy(LoopAttributes::DAC);

  EmitBlock(ConditionBlock);

  LoopStack.push(ConditionBlock, CGM.getContext(), CGM.getCodeGenOpts(), KokkosAttrs,
                 SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(Reattach, Reattach));

  // Create a scope for the condition variable cleanup.
  LexicalScope ConditionScope(*this, R);

  // Create the conditional.
  // DWS fix later
  EmitKokkosParallelForCond(InductionVarDecl, BE);
  //EmitKokkosParallelForCond(BE, InductionVarDecl, Detach, End, Sync);

  if (PForScope.requiresCleanups()) {
    EmitBlock(Cleanup.getBlock());
    EmitBranchThroughCleanup(Sync);
  }

  // Handle the detach block...
  EmitBlock(Detach);

  auto OldAllocaInsertPt = AllocaInsertPt;
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "", PForBody);

  llvm::Value *GInductionVar = GetAddrOfLocalVar(InductionVarDecl).getPointer();
  llvm::Value *GInductionVal = Builder.CreateLoad(GetAddrOfLocalVar(InductionVarDecl));

  QualType RefType = InductionVarDecl->getType();

  // Create the detach terminator
  Builder.CreateDetach(PForBody, Increment, SRStart);

  EmitBlock(PForBody);
  incrementProfileCounter(CE);

  llvm::AllocaInst *TLInductionVar =
      Builder.CreateAlloca(getTypes().ConvertType(RefType), nullptr,
                           InductionVarDecl->getName() + ".detach");
  Builder.CreateAlignedStore(GInductionVal, TLInductionVar,
                             getContext().getTypeAlignInChars(RefType));
                             
  std::pair<llvm::Value*, llvm::AllocaInst*> pair(GInductionVar, TLInductionVar);
  TLIVarList.push_back(pair);
  {
    if (BoundsList.size() == 0) {
      // Create a separate cleanup scope for the body, in case it is not
      // a compound statement.
      InKokkosConstruct = true;
      RunCleanupsScope BodyScope(*this);
      EmitStmt(Lambda->getBody());
      InKokkosConstruct = false;
      
      // Modify the body to use the ''detach''-local induction variable.
      // At this point in the codegen, the body block has been emitted 
      // and we can safely replace the ''sequential`` induction variable 
      // within the detach basic block.
      //
      // When Kokkos::Views (and likely some other structures) are used, we end up with extra blocks
      // between the original for loop body block and the reattach block. Without the loop, it will
      // only modify the last of these block. The loop iterates and updates all the blocks back to the
      // original for loop body block to use the thread-local induction variables
      //
      llvm::BasicBlock *CurrentBlock = Builder.GetInsertBlock();
      for (;;) {
        for (unsigned int i = 0; i<TLIVarList.size(); i++) {
          auto TLVar = TLIVarList.at(i).second;
          auto GInductionVar = TLIVarList.at(i).first;
          
          for(llvm::Value::use_iterator UI = GInductionVar->use_begin(), UE = GInductionVar->use_end(); 
              UI != UE; ) {
            llvm::Use &U = *UI++;
            llvm::Instruction *I = cast<llvm::Instruction>(U.getUser());
            if (I->getParent() == CurrentBlock) 
              U.set(TLVar);
          }
        }
        
        if (CurrentBlock == PForBody) {
            break;
        } else if (CurrentBlock->hasNPredecessorsOrMore(1) && CurrentBlock->getPrevNode()) {
             CurrentBlock = CurrentBlock->getPrevNode();
        } else { 
             break; 
        }
      }
    } else {
      EmitKokkosInnerLoop(CE, Lambda, ConditionBlock, BoundsList, params, TLIVarList, KokkosAttrs);
    }
  }
  
  auto tmp = AllocaInsertPt; 
  AllocaInsertPt = OldAllocaInsertPt; 
  tmp->removeFromParent(); 

  EmitBlock(Reattach.getBlock());
  Builder.CreateReattach(Increment, SRStart);

  EmitBlock(Increment);
  llvm::Value *IncVal = Builder.CreateLoad(GetAddrOfLocalVar(InductionVarDecl));
  llvm::Value *One = llvm::ConstantInt::get(ConvertType(InductionVarDecl->getType()), 1);
  IncVal = Builder.CreateAdd(IncVal, One);
  Builder.CreateStore(IncVal, GetAddrOfLocalVar(InductionVarDecl));

  BreakContinueStack.pop_back();
  ConditionScope.ForceCleanup();
  EmitStopPoint(CE);

  EmitBranch(ConditionBlock);
  PForScope.ForceCleanup();
  LoopStack.pop();

  EmitBlock(Sync.getBlock());
  Builder.CreateSync(End, SRStart);
  EmitBlock(End, true);
  return true;
}

bool CodeGenFunction::EmitKokkosParallelReduce(const CallExpr *CE,
                    ArrayRef<const Attr *> Attrs) {
  DiagnosticsEngine &Diags = CGM.getDiags();
  Diags.Report(CE->getExprLoc(), diag::warn_kokkos_reduce_unsupported);
  return false;
}
