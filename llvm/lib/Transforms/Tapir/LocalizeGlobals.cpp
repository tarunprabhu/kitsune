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

#include <functional>
#include <map>
#include <set>

namespace llvm {

static std::set<GlobalVariable*>&
collectUsedGlobals(Constant &c, std::set<GlobalVariable*> &seen);

static std::set<GlobalVariable*>&
collectUsedGlobals(BasicBlock &bb, std::set<GlobalVariable*> &seen) {
  for (Instruction &inst : bb)
    for (Use* op : inst.operands())
      if (Constant* c = dyn_cast<Constant>(&op))
        collectUsedGlobals(*c, seen);
  return seen;
}

static std::set<GlobalVariable*>&
collectUsedGlobals(Function &f, std::set<GlobalVariable*> &seen) {
  for (auto &bb : f)
    collectUsedGlobals(bb, seen);
  return seen;
}

static std::set<GlobalVariable*>&
collectUsedGlobals(GlobalVariable &g, std::set<GlobalVariable*> &seen) {
  seen.insert(&g);

  // No need to examine the initializer of the global to try and find any
  // global variables there. We are only concerned with those global variables
  // that are directly used by a function.
}

static std::set<GlobalVariable*> &
collectUsdGlobals(BlockAddress &blkaddr, std::set<GlobalVariable*> &seen) {
  if (Function *f = blkaddr.getFunction())
    collectUsedGlobals(*f, seen);
  if (BasicBlock*bb = blkaddr.getBasicBlock())
    collectUsedGlobals(*bb, seen);
  return seen;
}

static std::set<GlobalVariable*>&
collectUsedGlobals(Constant &c, std::set<GlobalVariable*> &seen) {
  if (GlobalVariable *g = dyn_cast<GlobalVariable>(&c))
    if (seen.find(g) != seen.end())
      return seen;

  if (Function* f = dyn_cast<Function>(&c))
    return collectUsedGlobals(*f, seen);
  else if (GlobalVariable* g = dyn_cast<GlobalVariable>(&c))
    return collectUsedGlobals(*g, seen);
  else if (BlockAddress* blkaddr = dyn_cast<BlockAddress>(&c))
    return collectUsedGlobals(*blkaddr, seen);
  else
    for (Use* op : c.operands())
      if (Constant* cop = dyn_cast<Constant>(op))
        collectUsedGlobals(*cop, seen);
  return seen;
}

static std::vector<GlobalVariable*> collectUsedGlobals(Function& F) {
  std::set<GlobalVariable*> usedGlobals;
  collectUsedGlobals(F, usedGlobals);

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

// Check if the given type is localizable. Pointer type are not localizable.
// Static arrays and vector types are localizable only if their element type
// is localizable. Structs are localizable only if all the element types are
// localizable.
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

// Check if the user of the given Use is in the given Function.
static bool isInFunction(Function &F, Use& U) {
  if (Instruction *I = dyn_cast<Instruction>(U.getUser()))
    return I->getParent()->getParent() == &F;
  return false;
}

// Convert all ConstantExpr's in the function to Instructions.
static bool constantExprsToInstructions(Function &F) {
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

// The implementation class that localizes the globals used by a single
// function.
class LocalizeGlobalsImpl {
public:
  using GlobalVariables = std::vector<GlobalVariable*>;

private:
  // The localization mode to use.
  LocalizeGlobals::Mode mode;

  // The function into which the local variables are being localized.
  Function& DeviceFunc;

  // The localized device function. This will be a clone of DeviceFunc with
  // potentially additional parameters for the global closures/globals that
  // are passed to the function as arguments
  Function* LocalizedFunc;

  // The global variables used in each function that are not locally constant
  // to that function.
  GlobalVariables ConstGlobals;

  // The global variables used in each function that are not locally constant
  // to that function.
  GlobalVariables NonConstGlobals;

  // The struct types for each function that will contain all the global
  // variables that have been localized. These will be null if the localization
  // mode is individual.
  StructType* ConstClosureType;
  StructType* NonConstClosureType;

  // The arguments in the localized function that correspond to the global
  // closures for const and non-const globals. If the localization mode is
  // individual, this will be null.
  StructType* ConstClosureArg;
  StructType* NonConstClosureArg;

private:
  void createClosureType(bool isConst) {
    std::vector<Type *> Elems;
    for (GlobalVariable* GV : isConst ? ConstGlobals : NonConstGlobals)
      Elems.push_back(GV->getValueType());

    std::string Name = F.getName().str() + ".global.closure";
    if (isConst) {
      return ConstClosureType =
        StructType::create(Elems, Name + ".const", false);
    } else {
      return NonConstClosureType =
        StructType::create(Elems, Name + ".nonconst", false);
    }
  }

  void LocalizeGlobals::cloneFunction() {
    Module &M =*OldF.getParent();

    FunctionType* OldFTy = DeviceFunc.getFunctionType();
    Type* ReturnTy = OldFTy->getReturnType();

    std::vector<Type*> Params(OldFTy->param_begin(), OldFTy->param_end());
    switch (mode) {
    case LocalizeGlobals::ValueStruct:
      Params.push_back(ConstClosureType);
      Params.push_back(NonConstClosureType->getPointerTo());
      break;
    case LocalizeGlobals::RefStruct:
      Params.push_back(ConstClosureType->getPointerTo());
      Params.push_back(NonConstClosureType->getPointerTo());
      break;
    case LocalizeGlobals::Individual:
      // Pass constant global variables by value. Some of these may be structs
      // but that is ok. At some point, we may consider something more clever
      // like passing only those fields of the struct that are used, but that
      // might get even more tricky.
      for (GlobalVariable* GV : ConstGlobals)
        Params.push_back(GV->getValueType());

      // Non-const structs are passed by pointer because their value needs to
      // be updated. The global variable will be copied to a variable on the
      // stack and that stack variable will be passed to the kernel.
      for (GlobalVariable* GV : NonConstGlobals)
        Params.push_back(GV->getType());
    }

    FunctionType* NewFTy = FunctionType::get(ReturnTy, Params, OldF.isVarArg());

    // The suffix added to the function name must still be legal in case
    // we need a valid runtime symbol to link against.
    LocalizedFunc =
        Function::Create(NewFTy, DeviceFunc.getLinkage(), 0,
                         DeviceFunc.getName() + "_with_closure", &M);

    // The only values that need to be mapped are the function arguments.
    ValueToValueMapTy VMap;
    for (Argument &OldArg : DeviceFunc.args()) {
      unsigned ArgNo = DeviceFunc.getArgNo();
      Argument &NewArg = *LocalizedFunc->getArg(ArgNo);
      NewArg.setName(DeviceFunc.getName());
      VMap[&OldArg] = &NewArg;
    }

    // It may be nice to name the additional arguments that were added, but
    // it's too much of a hassle and there's every chance that LLVM will get
    // rid of them anyway.

    SmallVector<ReturnInst*, 8> Returns;
    CloneFunctionInto(LocalizedFunc, &DeviceFunc, VMap,
                      CloneFunctionChangeType::LocalChangesOnly, Returns);

    if ((mode == LocalizeGlobals::ValueStruct)
        or (mode == LocalizeGlobals::RefStruct)) {
      ConstClosureArg = LocalizedFunc->getArg(Params.size() - 2);
      NonConstClosureArg = LocalizedFunc->getArg(Params.size() - 1);
    }
  }

  // Determine whether the global variables used by the function are locally
  // constant within the function.
  bool classifyUsedGlobals() {
    // FIXME: Do something cleverer than just assuming that all global variables
    // used by the function have to be declared constant to be locally constant.
    // Non-constant globals can be constant for the duration of a function call.
    for (GlobalVariable* GV : collectUsedGlobals(DeviceFunc))
      if (GV->isConstant())
        ConstGlobals.push_back(GV);
      else
        NonConstGlobals.push_back(GV);

    return ConstGlobals.size() or NonConstGlobals.size();
  }

  bool ensureLocalizableGlobals(const GlobalVariables& GVS) {
    for (GlobalVariable*GV : GVS) {
      if (not isLocalizableTy(GV->getValueType())) {
        // XXX: This should not assert out this way but should fail gracefully
        // but that would need support from the caller to do the right thing.
        // Right now, that isn't there, so just collapse.
        assert(0 && "Global variable used in Cuda kernel does not have "
               "localizable type");
        return false;
      }
    }
    return true;
  }

  size_t totalUsedGlobalSize() {
    size_t size = 0;
    const DataLayout& DL = DeviceFunc.getParent()->getDataLayout();

    for (Argument& Arg : DeviceFunc.args())
      size += DL.getTypeAllocSize(Arg.getType());

    for (GlobalVariable* GV : ConstGlobals)
      size += DL.getTypeAllocSize(GV->getValueType());

    for (GlobalVariable* GV : NonConstGlobals)
      size += DL.getTypeAllocSize(GV->getValueType());

    return size;
  }

  void preProcessDeviceFunction() {
    // Determine whether the global variables used in the function are
    // locally constant. If no global variables are used in the function,
    // terminate quickly.
    if (not classifyUsedGlobals())
      return;

    // If any of the global types is not localizable, check the return value
    // and do something sensible.
    ensureLocalizableGlobals(ConstGlobals);
    ensureLocalizableGlobals(NonConstGlobals);

    // FIXME: Add a check to ensure that this is only applicable to Cuda.
    //
    // FIXME: Do not hard-code the value of 4096.
    //
    // There is a maximum size for the parameters passed to a kernel. If the
    // total size of the used global variables exceeds this, for everything
    // to be passed by reference struct.
    if (totalUsedGlobalsSize() > 4096)
      mode = LocalizeGlobals::RefStruct;

    // The same ConstantExpr that uses a global may be shared among several
    // functions in a module. This makes it impossible to replace uses of a
    // global variable within a single function. Therefore, convert all
    // ConstantExpr's within the function to Instruction's. Instructions are
    // always associated with a single function.
    constantExprToInstruction(OldF);

    if (mode == LocalizeGlobals::ValuesStruct or
        mode == LocalizeGlobals::RefStruct) {
      createClosureType(ConstGVS, true);
      createClosureType(NonConstGVS, false);
    }

    // Adding an argument to a function is tricky. It is safest to clone
    // the function instead. We could potentially keep both this and the
    // original function around in case we need to make a decision at runtime
    // about which one to keep. All subsequent changes will be made to the
    // new function.
    cloneFunction();
  }

  // Localize the globals passed by value struct in the device.
  bool localizeGlobalsByValueStruct(Argument &ClosureArg,
                                    const GlobalVariables &GVS) {
    // Because the input to this code will be functions that have been
    // outlined by Tapir, they are guaranteed to have an entry block before
    // the loop. All instructions must be inserted before the first instruction
    // in the entry block.
    BasicBlock &Entry = LocalizedFunc->getEntryBlock();
    Instruction &FirstInst =*Entry.begin();

    for (size_t Idx = 0; Idx < GVS.size(); Idx++) {
      GlobalVariable* GV = GVS[Idx];

      ExtractValueInst *Val =
        ExtractValueInst::Create(&ClosureArg, Idx, "", &FirstInst);

      // Only those uses which are within the current function must be replaced.
      GV->replaceUsesWithIf(Val, std::bind(isInFunction, *LocalizedFunc, _1));
    }

    return GVS.size();
  }

  // Process the globals passed by ref struct in the device.
  // If the total size of the globals that need to be passed to the device
  // function is too large, then it may be passed by reference. The additional
  // parameter is provided in case the global variables are constant, in which
  // case there is no need to write the value of the global back.
  bool localizeGlobalsByRefStruct(Argument &ClosureArg,
                                  const GlobalVariables &GVS,
                                  bool isConst) {
    LLVMContext &Ctxt = DeviceModule.getContext();
    Type *Int32Ty = Type::getInt32Ty(Ctxt);
    Constant *Zero32 = Constant::getNullValue(Int32Ty);

    // Because the input to this code will be functions that have been
    // outlined by Tapir, they are guaranteed to have an entry block before
    // the loop. This is a very nice place to add any alloca instructions.
    BasicBlock &Entry = LocalizedFunc->getEntryBlock();
    BasicBlock &Exit = getUniqueExitBlock(F);
    Instruction &FirstInst =*Entry.begin();
    Instruction* Terminator = Exit.getTerminator();

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

      // At the end of the function, copy the values back out if the globals
      // are not locally constant.
      if (not isConst) {
        // Calculate the GEP again because it is best not to introduce a use
        // on the other end of the function from the def. Let the optimizer
        // decide if that is what it wants to do.
        GetElementPtrInst *GPtrOut
          = GetElementPtrInst::CreateInBounds(ClosureType, &ClosureArg,
                                              GPtrIndices, "", &FirstInst);
        LoadInst* OutVal = new LoadInst(ValueTy, Alloca, "", Terminator);
        new StoreInst(OutVal, GPtrOut, Terminator);
      }

      // The alloca instruction is a pointer to local memory. A global
      // variable is a pointer to global memory. Since there is this
      // equivalence, it is safe to replace all uses of the global with the
      // alloca. But only those uses which are within the current function
      // must be replaced.
      GV->replaceUsesWithIf(Alloca,
                            std::bind(isInFunction, *LocalizedFunc, _1));
    }
    return GVS.size();
  }

  bool localizeGlobalsByValueStruct() {
    bool changed = false;

    changed |= localizeGlobalsByValueStruct(*ConstClosureArg, ConstGVS);
    changed |= localizesGlobalsByRefStruct(*NonConstClosureArg, NonConstGVS,
                                           false);

    return changed;
  }

  bool localizeGlobalsByRefStruct() {
    bool changed = false;

    changed |= localizeGlobalsByRefStruct(*ConstClosureArg, ConstGVS, true);
    changed |= localizeGlobalsByRefStruct(*NonConstClosureArg, NonConstGVS,
                                          false);

    return changed;
  }

  bool localizeGlobalsIndividually() {
    unsigned FirstConstArgNo = DeviceFunc.arg_size();
    unsigned FirstNonConstArgNo = FirstConstArgNo + ConstGlobals.size();

    // Because the input to this code will be functions that have been
    // outlined by Tapir, they are guaranteed to have an entry block before
    // the loop. This is a very nice place to add any alloca instructions.
    BasicBlock &Entry = LocalizedFunc->getEntryBlock();
    BasicBlock &Exit = getUniqueExitBlock(F);
    Instruction &FirstInst =*Entry.begin();
    Instruction* Terminator = Exit.getTerminator();

    for (unsigned Idx = 0; Idx < ConstGlobals.size(); Idx++) {
      GlobalVariable* GV = ConstGlobals[Idx];
      Type* ValTy = GV->getValueType();
      Argument& Arg = *LocalizedFunc->getArg(FirstConstArgNo + Idx);

      // Create a local copy for the value because the global variables are
      // assumed to be pointers, so they will need an address from which to
      // load the value.
      AllocaInst* LocalCopy = new AllocaInst(ValTy, "", &FirstInst);
      new StoreInst(Arg, LocalCopy, &FirstInst);

      GV->replaceUsesWithIf(LocalCopy,
                            std::bind(isInFunction, *LocalizedFunc, _1));
    }

    for (unsigned Idx = 0; Idx < NonConstGlobals.size(); Idx++) {
      GlobalVariable* GV = ConstGlobals[Idx];
      Type* ValTy = GV->getValueType();
      Argument& Arg = *LocalizedFunc->getArg(FirstNonConstArgNo + Idx);

      GV->replaceUsesWithIf(LocalCopy,
                            std::bind(isInFunction, *LocalizedFunc, _1));
    }

    return ConstantGlobals.size() or NonConstGlobals.size();
  }

  // Localize the globals in the given device function. Return true if at
  // least one global variable was localized, false otherwise.
  bool localizeGlobalsInDeviceFunction() {
    switch (mode) {
    case LocalizeGlobals::ValueStruct:
      return localizeGlobalsByValueStruct();
    case LocalizeGlobals::RefStruct:
      return localizeGlobalsByRefStruct();
    case LocalizeGlobals::Individually:
      return localizeGlobalsIndividually();
    default:
      assert(0 && "Unknown localization mode");
      break;
    }
    return false;
  }

  void passGlobalsByValueStruct(CallBase& Call,
                                const GlobalVariables& GVS,
                                StructType* ClosureTy,
                                std::vector<Value*>& NewCallArgs) {
    LLVMContext &Ctxt = Call.getContext();
    Type *Int32Ty = Type::getInt32Ty(Ctxt);
    Constant *Zero32 = Constant::getNullValue(Int32Ty);

    Value* Closure = UndefValue::get(ClosureTy);
    if (GVS.size()) {
      for (unsigned Idx = 0; Idx < GVS.size(); Idx++) {
        GlobalVariable* GV = GVS[i];
        LoadInst* GVal = new LoadInst(GV->getValueTy(), GV, "", &Call);
        Closure = InsertValueInst::Create(Closure, GVal, Idx, "", &Call);
      }
    }
    NewCallArgs.push_back(Closure);
  }

  void passGlobalsByRefStruct(CallBase& Call,
                              const GlobalVariables& GVS,
                              StructType* ClosureTy,
                              bool isConst,
                              std::vector<Value*>& NewCallArgs) {
    LLVMContext &Ctxt = Call.getContext();
    Type *Int32Ty = Type::getInt32Ty(Ctxt);
    Constant *Zero32 = Constant::getNullValue(Int32Ty);

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
      if (not isConst) {
        Instruction* AfterCall = Call.getNextNonDebugInstruction();
        GetElementPtrInst* GPtrOut
          = GetElementPtrInst::CreateInBounds(ClosureTy, ClosureArg, GPtrIndices,
                                              "", AfterCall);
        LoadInst* OutVal = new LoadInst(ValueTy, GPtrOut, "", AfterCall);
        new StoreInst(OutVal, GV, AfterCall);
      }
    }
    NewCallArgs.push_back(ClosureArg);
  }

  void passGlobalsIndividually(CallBase& Call,
                               const GlobalVariables& GVS,
                               bool isConst,
                               std::vector<Value*>& NewCallArgs) {
    if (isConst) {
      for (GlobalVariable* GV : GVS)
        NewCallArgs.push_back(new LoadInst(GV->getValueType(), GV, "", &Call));
    } else {
      for (GlobalVariable* GV : GVS) {
        // Local copy of the global variable. As I understand it, pointers to
        // global variables cannot be passed directly to a Cuda kernel, but a
        // pointer to a local variable can.
        Type* ValueTy = GV->getValueType();

        // Create a local copy of the global.
        AllocaInst* LocalCopy = new AllocaInst(ValueTy, 0, "", &Call);
        LoadInst* InVal = new LoadInst(ValueTy, GV, "", &Call);
        new StoreInst(InVal, LocalCopy, &Call);

        // Update the global value after the kernel returns.
        Instruction* AfterCall = Call.getNextNonDebugInstruction();
        LoadInst* OutVal = new LoadInst(ValueTy, LocalCopy, "", AfterCall);
        new StoreInst(OutVal, GV, AfterCall);

        NewCallArgs.push_back(LocalCopy);
      }
    }
  }

  void fixCallToLocalizedFunction(CallBase &Call,
                                  const GlobalVariables& HostConstGlobals,
                                  const GlobalVariables& HostNonConstGlobals) {
    std::vector<Value*> NewCallArgs(Call.arg_begin(), Call.arg_end());

    switch (mode) {
    case LocalizeMode::ValueStruct:
      passGlobalsByValueStruct(Call, HostConstGlobals, ConstClosureType,
                               NewCallArgs);
      passGlobalsByRefStruct(Call, HostNonConstGlobals, NonConstClosureType,
                             false, NewCallArgs);
      break;
    case LocalizeMode::RefStruct:
      passGlobalsByRefStruct(Call, HostConstGlobals, ConstClosureType, true,
                             NewCAllArgs);
      passGlobalsByRefStruct(Call, HostNonConstGlobals, NonConstClosureType,
                             false, NewCallArgs);
      break;
    case LocalizeMode::Individually:
      passGlobalsIndividually(Call, HostConstGlobals, true, NewCallArgs);
      passGlobalsIndividually(Call, HostNonConstGlobals, false, NewCallArgs);
      break;
    default:
      assert(0 && "Unknown localization mode.");
      break;
    }

    // Replace call.
    Function* LocalizedFunc = localizedFuncs.at(&DeviceF);
    CallInst* NewCall = CallInst::Create(LocalizedFunc->getFunctionType(),
                                         LocalizedFunc,
                                         NewCallArgs,
                                         "",
                                         &Call);
    Call.replaceAllUsesWith(NewCall);
    Call.removeFromParent();
  }

  // Modify the callers of the given device function in the given host module
  // to pass a global closure. This assumes that the globals in the device
  // function have already been localized. If the device function does not
  // access any global variables, the host module will be unchanged. Returns
  // true if the host module was changed.
  bool fixCallsToLocalizedFunction(
      Module &HostModule,
      const LocalizeGlobals::DeviceToHostMap &deviceToHostMap) {
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

    GlobalVariables HostConstGlobals;
    for (GlobalVariable *GV : ConstGlobals) {
      auto &It = deviceToHostMap.find(GV);
      if (It != deviceToHostMap.end())
        HostConstGlobals.push_back(*It);
      else
        HostConstGlobals.push_back(GV);
    }

    GlobalVariables HostNonConstGlobals;
    for (GlobalVariable *GV : NonConstGlobals) {
      auto &It = deviceToHostMap.find(GV);
      if (It != deviceToHostMap.end())
        HostNonConstGlobals.push_back(*It);
      else
        HostNonConstGlobals.push_back(GV);
    }

    for (CallBase* Call : Calls)
      fixCallToLocalizedFunction(*Call, HostConstGlobals, HostNonConstGlobals);

    return Calls.size();
  }

  // Main entry point for the class. Will localize global variables in the
  // device function and fix all calls in the host. If no globals
  // were localized, both the host and device modules are left unchanged.
  bool run(Module& HostModule,
           const LocalizeGlobals::DeviceToHostMap& deviceToHostMap) {
    bool changed = false;
    if (preProcessDeviceFunction()) {
      localizeGlobalsInDeviceFunction();
      changed |= fixCallsToLocalizedFunction(HostModule);
    }
    return changed;
  }

public:
  LocalizeGlobalImpl(LocalizeGlobals::Mode mode, Function &DeviceFunc)
      : mode(mode), DeviceFunc(DeviceFunc), LocalizedFunc(nullptr),
        ConstClosureType(nullptr), NonConstClosureType(nullptr),
        ConstClosureArg(nullptr), NonConstClosureArg(nullptr) {
    // This should not actually happen because all the functions here will
    // have been created from outlined loops, but just in case that changes,
    // don't support vararg functions.
    assert(!DeviceFunc.isVarArg() &&
           "VarArg functions not supported as cuda kernels");
  }
};

LocalizeGlobals::LocalizeGlobals(Module &DeviceModule, Mode mode,
                                 const DeviceToHostMap &deviceToHostMap)
    : DeviceModule(DeviceModule), mode(mode), deviceToHostMap(deviceToHostMap) {
  ;
}

bool LocalizeGlobals::localizeGlobalsInDeviceFunction(Function &DeviceFunc,
                                                      Module& HostModule) {
  assert((DeviceFunc.getParent() == &DeviceModule)
         && "Function being localized must be in the device module.");

  LocalizeGlobalsImpl localizer(mode, DeviceFunc);
  return localizer.run(HostModule, deviceToHostMap);
}

} // namespace llvm
