//===- CudaLocalizeGlobalsPass.cpp - Localize globals in Cuda kernels -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that groups all the global variables used by
// each function in the module in to a single struct and passes that struct
// explicitly as an additional argument to the function.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalIFunc.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Transforms/Tapir/LocalizeGlobals.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <map>
#include <set>

namespace llvm {

static std::set<GlobalVariable*> &collect(Constant &c,
                                          std::set<GlobalVariable*> &seen);

static std::set<GlobalVariable*> &collect(BasicBlock &bb,
                                          std::set<GlobalVariable*> &seen) {
  for (auto &inst : bb)
    for (auto &op : inst.operands())
      if (auto*c = dyn_cast<Constant>(&op))
        collect(*c, seen);
  return seen;
}

static std::set<GlobalVariable*> &collect(Function &f,
                                          std::set<GlobalVariable*> &seen) {
  for (auto &bb : f)
    collect(bb, seen);
  return seen;
}

static std::set<GlobalVariable*> &collect(GlobalVariable &g,
                                          std::set<GlobalVariable*> &seen) {
  seen.insert(&g);

  if (g.hasInitializer())
    collect(*g.getInitializer(), seen);
  return seen;
}

static std::set<GlobalVariable*> &collect(BlockAddress &blkaddr,
                                          std::set<GlobalVariable*> &seen) {
  if (Function*f = blkaddr.getFunction())
    collect(*f, seen);
  if (BasicBlock*bb = blkaddr.getBasicBlock())
    collect(*bb, seen);
  return seen;
}

static std::set<GlobalVariable*> &collect(Constant &c,
                                          std::set<GlobalVariable*> &seen) {
  if (GlobalVariable*g = dyn_cast<GlobalVariable>(&c))
    if (seen.find(g) != seen.end())
      return seen;

  if (auto*f = dyn_cast<Function>(&c))
    return collect(*f, seen);
  else if (auto*g = dyn_cast<GlobalVariable>(&c))
    return collect(*g, seen);
  else if (auto*blkaddr = dyn_cast<BlockAddress>(&c))
    return collect(*blkaddr, seen);
  else
    for (auto &op : c.operands())
      if (auto*cop = dyn_cast<Constant>(op))
        collect(*cop, seen);
  return seen;
}

static std::vector<GlobalVariable*> collect(Function& F) {
  std::set<GlobalVariable*> usedGlobals;
  collect(F, usedGlobals);

  return std::vector<GlobalVariable*>(usedGlobals.begin(), usedGlobals.end());
}

// Ideally, this should never fail because one of the standard passes that
// runs ensures that every function has a unique exit block. Unfortunately,
// the global localizer is not currently a pass, so there's no reasonable way
// to ensure that that pass is run.
static BasicBlock& getUniqueExitBlock(Function& F) {
  std::vector<BasicBlock*> ExitingBlocks;
  for (BasicBlock& BB : F) {
    Instruction* Term = BB.getTerminator();
    if (isa<ReturnInst>(Term))
      ExitingBlocks.push_back(&BB);
    else if (isa<UnreachableInst>(Term))
      // This is kept separate because there may be a way of reasonably
      // dealing with it - maybe by just treating it as a return instruction.
      assert(0 && "Unsupported instruction when localizing globals "
                  "(UnreachableInst)");
    else if (isa<CatchReturnInst>(Term) or isa<CatchSwitchInst>(Term) or
             isa<CleanupReturnInst>(Term))
      assert(0 && "Unsupported instruction when localizing globals "
                  "(CatReturnInst || CatchSwitchINst || CleanupReturnInst)");
  }

  assert(ExitingBlocks.size() == 1 &&
         "Function must have a unique exit block.");
  return *ExitingBlocks.front();
}

// Recursively descend into the first operand of the Instruction or ConstantExpr
// and get the innermost value.
static Value* getInnermostOperand(Value* V) {
  if (Instruction* Inst = dyn_cast<Instruction>(V))
    return getInnermostOperand(*Inst->op_begin());
  else if (ConstantExpr* CExpr = dyn_cast<ConstantExpr>(V))
    return getInnermostOperand(*CExpr->op_begin());
  return V;
}

static Function* getCalledFunction(CallBase& Call) {
  // If a function is used without a cast, getCalledFunction() will return it.
  // Otherwise, it will be a BitcastConstantExpr, so we need to pull the
  // function out from within it. It is unlikely that there will be any other
  // cases, but just in case, try those at the end.
  if (Function* Callee = Call.getCalledFunction())
    return Callee;
  else if (auto* CExpr = dyn_cast<ConstantExpr>(Call.getCalledOperand()))
    if (CExpr->getNumOperands() == 1)
      if (Function* Callee = dyn_cast<Function>(&*CExpr->op_begin()))
        return Callee;
  return dyn_cast<Function>(getInnermostOperand(Call.getCalledOperand()));
}

static bool isLocalizableTy(Type *Ty);

static bool isLocalizableTy(StructType *StructTy) {
  for (Type *ElemTy : StructTy->elements())
    if (not isLocalizableTy(ElemTy))
      return false;
  return true;
}

static bool isLocalizableTy(Type *Ty) {
  if (Ty->isIntegerTy() or Ty->isHalfTy() or Ty->isFloatingPointTy())
    return true;
  else if (ArrayType *ArrayTy = dyn_cast<ArrayType>(Ty))
    return isLocalizableTy(ArrayTy->getElementType());
  else if (VectorType *VecTy = dyn_cast<VectorType>(Ty))
    return isLocalizableTy(VecTy->getElementType());
  else if (StructType *StructTy = dyn_cast<StructType>(Ty))
    return isLocalizableTy(StructTy);
  else
    return false;
}

LocalizeGlobals::LocalizeGlobals(Module& DeviceModule,
                                 const DeviceToHostMap& deviceToHostMap)
  : DeviceModule(DeviceModule), deviceToHostMap(deviceToHostMap) {
  ;
}

std::vector<GlobalVariable*>
LocalizeGlobals::getHostGlobalsUsedByDeviceFunction(Function &F) {
  std::vector<GlobalVariable*> usedHostGlobals;
  for (GlobalVariable *GV : usedGlobals.at(&F))
    usedHostGlobals.push_back(deviceToHostMap.at(GV));
  return usedHostGlobals;
}

bool LocalizeGlobals::constantExprToInstruction(Function &F) {
  bool replaced = false;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      for (unsigned op = 0; op < I.getNumOperands(); op++) {
        if (ConstantExpr*C = dyn_cast<ConstantExpr>(I.getOperand(op))) {
          Instruction*NewI = C->getAsInstruction();
          NewI->insertBefore(&I);
          I.setOperand(op, NewI);
          replaced = true;
        }
      }
    }
  }
  return replaced;
}

Function& LocalizeGlobals::cloneFunction(Function &OldF,
                                         StructType*ClosureType) {
  Module &M =*OldF.getParent();

  FunctionType*OldFTy = OldF.getFunctionType();
  std::vector<Type*> Params(OldFTy->param_begin(), OldFTy->param_end());
  Params.push_back(ClosureType->getPointerTo());
  FunctionType*NewFTy =
    FunctionType::get(OldFTy->getReturnType(), Params, OldF.isVarArg());

  // The suffix added to the function name must still be legal in case
  // we need a valid runtime symbol to link against.
  Function &F =*Function::Create(NewFTy, OldF.getLinkage(), 0,
                                 OldF.getName() + "_with_closure", &M);

  // The only values that need to be mapped are the function arguments.
  ValueToValueMapTy VMap;
  for (Argument &OldArg : OldF.args()) {
    unsigned ArgNo = OldArg.getArgNo();
    Argument &NewArg =*F.getArg(ArgNo);
    NewArg.setName(OldArg.getName());
    VMap[&OldArg] = &NewArg;
  }
  Argument &ClosureArg =*F.getArg(Params.size() - 1);
  ClosureArg.setName("global.closure.arg");

  SmallVector<ReturnInst*, 8> Returns; // Ignored
  CloneFunctionInto(&F, &OldF, VMap,
                    CloneFunctionChangeType::LocalChangesOnly, Returns);

  localizedFuncs[&OldF] = &F;
  return *localizedFuncs.at(&OldF);
}

StructType* LocalizeGlobals::createClosureType(Function &F) {
  std::vector<Type*> ClosureElems;
  for (GlobalVariable*GV : usedGlobals.at(&F))
    ClosureElems.push_back(GV->getValueType());

  std::string ClosureName = F.getName().str() + ".global.closure";
  closureTypes[&F] = StructType::create(ClosureElems, ClosureName, false);

  return closureTypes.at(&F);
}

bool LocalizeGlobals::localizeGlobalsInDeviceFunction(Function &OldF) {
  assert((OldF.getParent() == &DeviceModule)
         && "Function being localized must be in the device module.");

  LLVMContext &Ctxt = DeviceModule.getContext();
  Type*Int32Ty = Type::getInt32Ty(Ctxt);
  Constant*Zero32 = Constant::getNullValue(Int32Ty);

  usedGlobals[&OldF] = collect(OldF);

  // Sanity check. This can only work if the global variables are either
  // scalars, or static arrays.
  for (const auto &GI : usedGlobals) {
    Function &F =*GI.first;

    // This should not actually happen because all the functions here will
    // have been created from outlined loops, but just in case that changes,
    // don't support vararg functions.
    assert(!F.isVarArg() && "VarArg functions not supported as cuda kernels");

    for (GlobalVariable*GV : GI.second) {
      if (not isLocalizableTy(GV->getValueType())) {
        // XXX: This should not assert out this way but should fail gracefully
        // but that would need support from the caller to do the right thing.
        // Right now, that isn't there, so just collapse.
        assert(0 && "Global variable used in Cuda kernel does not have "
               "localizable type");
        return false;
      }
    }
  }

  // The same ConstantExpr that uses a global may be shared among several
  // functions in a module. This makes it impossible to replace uses of a global
  // variable within a single function. Therefore, convert all ConstantExpr's
  // within the function to Instruction's. Instructions are always associated
  // with a single function.
  if (not usedGlobals.at(&OldF).empty())
    constantExprToInstruction(OldF);

  // Create a local variable for each of the globals. This ensures that they
  // will have an address that can be taken.
  for (auto &GI : usedGlobals) {
    Function &OldF =*GI.first;
    const std::vector<GlobalVariable*> &GVS = GI.second;

    // Create a closure type for the function.
    StructType*ClosureType = createClosureType(OldF);

    // Adding an argument to a function is tricky. It is safest to clone
    // the function instead. We could potentially keep both this and the
    // original function around in case we need to make a decision at runtime
    // about which one to keep. All subsequent changes will be made to the
    // new function.
    Function &F = cloneFunction(OldF, ClosureType);

    // arg_size() is guaranteed to be 1 because the closure arg will have
    // been added to a function with 0 arguments.
    Argument &ClosureArg =*F.getArg(F.arg_size() - 1);

    // This lambda determines whether the given Use is in the current
    // function.
    auto isInCurrentFunction = [&](Use &U) {
      if (Instruction*I = dyn_cast<Instruction>(U.getUser()))
        return I->getParent()->getParent() == &F;
      return false;
    };

    // Because the input to this code will be functions that have been
    // outlined by Tapir, they are guaranteed to have an entry block before
    // the loop. This is a very nice place to add any alloca instructions.
    BasicBlock &Entry = F.getEntryBlock();

    // The function should have a unique exit block
    BasicBlock &Exit = getUniqueExitBlock(F);
    Instruction* Terminator = Exit.getTerminator();

    // Keep track of the first instruction in the block before any
    // modifications are made. All instructions must be inserted before this
    Instruction &FirstInst =*Entry.begin();

    for (size_t Idx = 0; Idx < GVS.size(); Idx++) {
      GlobalVariable* GV = GVS[Idx];
      Type *ValueTy = GV->getValueType();

      // Local storage for the closure value corresponding to the global.
      AllocaInst *Alloca =
        new AllocaInst(ValueTy, 0, GV->getName() + ".local", &FirstInst);

      // Offset of this global variable in the closure argument.
      Constant *CIdx = ConstantInt::get(Int32Ty, Idx, false);
      Value *GPtrIndices[] = {Zero32, CIdx};

      // Copy values from the closure argument to local storage.
      GetElementPtrInst *GPtrIn
        = GetElementPtrInst::CreateInBounds(ClosureType, &ClosureArg,
                                            GPtrIndices, "", &FirstInst);
      LoadInst *InVal = new LoadInst(ValueTy, GPtrIn, "", &FirstInst);
      new StoreInst(InVal, Alloca, &FirstInst);

      // At the end of the function, copy the values back out.
      // Calculate the GEP again because it is best not to introduce that
      // spans the entire function. Repeating is ok because the optimizer will
      // get rid of it.
      GetElementPtrInst *GPtrOut
        = GetElementPtrInst::CreateInBounds(ClosureType, &ClosureArg,
                                            GPtrIndices, "", &FirstInst);
      LoadInst* OutVal = new LoadInst(ValueTy, Alloca, "", Terminator);
      new StoreInst(OutVal, GPtrOut, Terminator);

      // The alloca instruction is a pointer to local memory. A global
      // variable is a pointer to global memory. Since there is this
      // equivalence, it is safe to replace all uses of the global with the
      // alloca. But only those uses which are within the current function
      // must be replaced.
      GV->replaceUsesWithIf(Alloca, isInCurrentFunction);
    }
  }

  return not usedGlobals.at(&OldF).empty();
}

void LocalizeGlobals::fixCallToLocalizedFunction(CallBase &Call,
                                                 Function &DeviceF) {
  LLVMContext &Ctxt = Call.getContext();

  Type*Int32Ty = Type::getInt32Ty(Ctxt);
  Constant*Zero32 = Constant::getNullValue(Int32Ty);

  const std::vector<GlobalVariable*>& GVS =
    getHostGlobalsUsedByDeviceFunction(DeviceF);
  StructType* ClosureTy = closureTypes.at(&DeviceF);

  // Storage for the closure argument for this particular call-site.
  // There will be one for each callsite where the function is called.
  AllocaInst* ClosureArg = new AllocaInst(ClosureTy, 0, "", &Call);

  for (size_t Idx = 0; Idx < GVS.size(); Idx++) {
    GlobalVariable* GV = GVS[Idx];
    Type* ValueTy = GV->getValueType();

    // Offset of the global variable in the closure argument.
    Constant *CIdx = ConstantInt::get(Int32Ty, Idx, false);
    Value *GPtrIndices[] = {Zero32, CIdx};
    GetElementPtrInst* GPtrIn
      = GetElementPtrInst::CreateInBounds(ClosureTy, ClosureArg, GPtrIndices,
                                          "", &Call);

    // Save the value of the global variable into the closure argument.
    LoadInst* InVal = new LoadInst(ValueTy, GV, "", &Call);
    new StoreInst(InVal, GPtrIn, &Call);

    // Copy the value of the global back from the closure argument.
    // This should only be done if the global variable is not a constant.
    //
    // TODO?: It would be good to be able to determine using a static analysis
    // which globals are constant for the duration of the call to avoid having
    // to copy them in and out. Not sure how helpful that would be, but it
    // may be nice to have.
    if (not GV->isConstant()) {
      Instruction* AfterCall = Call.getNextNonDebugInstruction();
      GetElementPtrInst* GPtrOut
        = GetElementPtrInst::CreateInBounds(ClosureTy, ClosureArg, GPtrIndices,
                                            "", AfterCall);
      LoadInst* OutVal = new LoadInst(ValueTy, GPtrOut, "", AfterCall);
      new StoreInst(OutVal, GV, AfterCall);
    }
  }

  // Replace call.
  Function* LocalizedFunc = localizedFuncs.at(&DeviceF);
  std::vector<Value*> NewCallArgs(Call.arg_begin(), Call.arg_end());
  NewCallArgs.push_back(ClosureArg);
  CallInst* NewCall = CallInst::Create(LocalizedFunc->getFunctionType(),
                                       LocalizedFunc,
                                       NewCallArgs,
                                       "",
                                       &Call);
  Call.replaceAllUsesWith(NewCall);
  Call.removeFromParent();
}

bool LocalizeGlobals::fixCallsToLocalizedFunction(Function& DeviceF,
                                                  Module& HostModule) {
  // The calls cannot be replaced as they are encountered because that will
  // break the iteration over the instructions.
  std::vector<CallBase*> Calls;
  for (Function& F : HostModule)
    for (BasicBlock& BB : F)
      for (Instruction& I : BB)
        if (CallBase* Call = dyn_cast<CallBase>(&I))
          if (Function* Callee = getCalledFunction(*Call))
            if (Callee == &DeviceF)
              Calls.push_back(Call);

  for (CallBase* Call : Calls)
    fixCallToLocalizedFunction(*Call, DeviceF);

  return Calls.size();
}

} // namespace llvm
