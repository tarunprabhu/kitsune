//===- TapirGPUUtils.cpp - Lower Tapir to the Kitsune GPU back end --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Kitsune+Tapir HIP ABI to convert Tapir
// instructions to calls into the HIP-centric portions of the Kitsune
// runtime for GPUs to produce a fully compiled (not JIT) executable
// that is suitable for a given architecture target.
//
// NOTE: Several aspects of this transform mimic Clang's code generation
// for HIP. Any significant changes to Clang at that level might require
// changes here as well.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/TapirGPUUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"

#include <set>

namespace llvm {

namespace tapir {

static void collectGlobalValues(Constant &c, std::set<GlobalValue *> &seen);

static void collectGlobalValues(GlobalVariable &g,
                                std::set<GlobalValue *> &seen) {
  seen.insert(&g);
  if (g.hasInitializer())
    collectGlobalValues(*g.getInitializer(), seen);
}

static void collectGlobalValues(GlobalIFunc &g, std::set<GlobalValue *> &seen) {
  seen.insert(&g);
  llvm_unreachable("kitsune: GNU IFUNC not yet supported");
}

static void collectGlobalValues(GlobalAlias &g, std::set<GlobalValue *> &seen) {
  seen.insert(&g);
  llvm_unreachable("kitsune: GlobalAlias not yet supported");
}

static void collectGlobalValues(BlockAddress &blkaddr,
                                std::set<GlobalValue *> &seen) {
  if (Function *f = blkaddr.getFunction())
    collectGlobalValues(*f, seen);
  if (BasicBlock *bb = blkaddr.getBasicBlock())
    collectGlobalValues(*bb, seen);
}

static void collectGlobalValues(Constant &c, std::set<GlobalValue *> &seen) {
  if (GlobalValue *g = dyn_cast<GlobalValue>(&c))
    if (seen.find(g) != seen.end())
      return;

  if (auto *f = dyn_cast<Function>(&c))
    return collectGlobalValues(*f, seen);
  else if (auto *g = dyn_cast<GlobalVariable>(&c))
    return collectGlobalValues(*g, seen);
  else if (auto *g = dyn_cast<GlobalAlias>(&c))
    return collectGlobalValues(*g, seen);
  else if (auto *g = dyn_cast<GlobalIFunc>(&c))
    return collectGlobalValues(*g, seen);
  else if (auto *blkaddr = dyn_cast<BlockAddress>(&c))
    return collectGlobalValues(*blkaddr, seen);
  else
    for (Use &op : c.operands())
      if (auto *cop = dyn_cast<Constant>(op))
        collectGlobalValues(*cop, seen);
}

void collectGlobalValues(BasicBlock &bb, std::set<GlobalValue *> &seen) {
  for (Instruction &inst : bb)
    for (Use &op : inst.operands())
      if (auto *c = dyn_cast<Constant>(&op))
        collectGlobalValues(*c, seen);
}

void collectGlobalValues(Function &f, std::set<GlobalValue *> &seen) {
  seen.insert(&f);
  for (BasicBlock &bb : f)
    collectGlobalValues(bb, seen);
}

CodeGenOptLevel mapToCodeGenOptLevel(OptimizationLevel OptLevel) {
  switch (OptLevel.getSpeedupLevel()) {
  case 0:
    return CodeGenOptLevel::None;
  case 1:
    return CodeGenOptLevel::Less;
  case 2:
    return CodeGenOptLevel::Default;
  case 3:
    return CodeGenOptLevel::Aggressive;
  default:
    llvm_unreachable("mapToCodeGenOptLevel: unknown speedup level");
  }
}

OptimizationLevel mapToOptimizationLevel(unsigned OptLevel) {
  switch (OptLevel) {
  case 0:
    return OptimizationLevel::O0;
  case 1:
    return OptimizationLevel::O1;
  case 2:
    return OptimizationLevel::O2;
  case 3:
    return OptimizationLevel::O3;
  default:
    llvm_unreachable("mapToOptimizationLevel: invalid optimization level");
  }
}

raw_ostream &renderCommandLine(ArrayRef<StringRef> args, raw_ostream &os) {
  if (args.size()) {
    os << args.front();
    for (size_t i = 1; i < args.size(); ++i)
      os << " " << args[i];
    os << "\n";
  }
  return os;
}

Constant *getOrInsertFBGlobal(Module &m, StringRef name, Type *ty) {
  return m.getOrInsertGlobal(name, ty, [&] {
    LLVMContext &ctxt = m.getContext();
    PointerType *ptrTy = PointerType::getUnqual(ctxt);
    return new GlobalVariable(m, ty, true, GlobalValue::InternalLinkage,
                              ConstantPointerNull::get(ptrTy), name, nullptr);
  });
}

Constant *createConstantStr(const std::string &Str, Module &M,
                            const std::string &Name,
                            const std::string &SectionName,
                            unsigned Alignment) {
  LLVMContext &Ctx = M.getContext();
  Constant *CSN = ConstantDataArray::getString(Ctx, Str);
  GlobalVariable *GV = new GlobalVariable(
      M, CSN->getType(), true, GlobalVariable::PrivateLinkage, CSN, Name);
  Type *StrTy = GV->getType();

  const DataLayout &DL = M.getDataLayout();
  Constant *Zeros[] = {ConstantInt::get(DL.getIndexType(StrTy), 0),
                       ConstantInt::get(DL.getIndexType(StrTy), 0)};
  if (!SectionName.empty()) {
    GV->setSection(SectionName);
    // Mark the address as used which make sure that this section isn't
    // merged and we will really have it in the object file.
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
  }

  if (Alignment)
    GV->setAlignment(Align(Alignment));

  Constant *CS = ConstantExpr::getGetElementPtr(GV->getValueType(), GV, Zeros);
  return CS;
}

// Adapted from Transforms/Utils/ModuleUtils.cpp
void appendToGlobalCtors(Module &M, Constant *C, int Priority, Constant *Data) {
  IRBuilder<> IRB(M.getContext());
  FunctionType *FnTy = FunctionType::get(IRB.getVoidTy(), false);

  // Get the current set of static global constructors and add
  // the new ctor to the list.
  SmallVector<Constant *, 16> CurrentCtors;
  StructType *EltTy = StructType::get(
      IRB.getInt32Ty(), PointerType::getUnqual(FnTy), IRB.getPtrTy());
  if (GlobalVariable *GVCtor = M.getNamedGlobal("llvm.global_ctors")) {
    if (Constant *Init = GVCtor->getInitializer()) {
      unsigned N = Init->getNumOperands();
      CurrentCtors.reserve(N + 1);
      for (unsigned i = 0; i != N; ++i)
        CurrentCtors.push_back(cast<Constant>(Init->getOperand(i)));
    }
    GVCtor->eraseFromParent();
  }

  // Build a 3 field global_ctor entry.
  // We don't take a comdat key.
  Constant *CSVals[3];
  CSVals[0] = IRB.getInt32(Priority);
  CSVals[1] = C;
  CSVals[2] = Data ? ConstantExpr::getPointerCast(Data, IRB.getPtrTy())
                   : Constant::getNullValue(IRB.getPtrTy());
  Constant *RuntimeCtorInit = ConstantStruct::get(
      EltTy, ArrayRef<Constant *>(CSVals, EltTy->getNumElements()));

  CurrentCtors.push_back(RuntimeCtorInit);

  // Create a new initializer.
  ArrayType *AT = ArrayType::get(EltTy, CurrentCtors.size());
  Constant *NewInit = ConstantArray::get(AT, CurrentCtors);

  // Create the new global variable and replace all uses of
  // the old global variable with the new one.
  (void)new GlobalVariable(M, NewInit->getType(), false,
                           GlobalValue::AppendingLinkage, NewInit,
                           "llvm.global_ctors");
}

KernelInstMixData getKernelInstructionMix(const Function &f) {
  KernelInstMixData instMix;

  std::set<const Function *> calledFuncs;
  for (const_inst_iterator i = inst_begin(f); i != inst_end(f); ++i) {
    if (i->mayReadOrWriteMemory()) {
      instMix.numMemoryOps++;
    } else if (i->isUnaryOp() or i->isBinaryOp()) {
      if (i->getType()->isFPOrFPVectorTy())
        instMix.numFlops++;
      else if (i->getType()->isIntegerTy())
        instMix.numIntOps++;
      else
        instMix.numOtherOps++;
    } else if (auto *call = dyn_cast<CallInst>(&*i)) {
      calledFuncs.insert(call->getCalledFunction());
    }
  }

  for (const Function *called : calledFuncs) {
    KernelInstMixData localInstMix = getKernelInstructionMix(*called);
    instMix.numMemoryOps += localInstMix.numMemoryOps;
    instMix.numFlops += localInstMix.numFlops;
    instMix.numIntOps += localInstMix.numIntOps;
    instMix.numOtherOps += localInstMix.numOtherOps;
  }

  return instMix;
}

} // namespace tapir

} // namespace llvm
