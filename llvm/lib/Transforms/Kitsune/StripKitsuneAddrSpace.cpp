//==- StripKitsuneAddrSpace.cpp - Strip Kitsune's address spaces -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the KitsuneStripAddrSpacePass. This mutates type with any
// kitsune-specific address space and puts those pointers into the default
// address space instead.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Kitsune/StripKitsuneAddrSpace.h"
#include "kitsune/Config/config.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include <map>
#include <set>

using namespace llvm;

#define DEBUG_TYPE "strip-kitsune-addrspace"

class StripKitsuneAddrSpaceImpl {
private:
  /// The functions whose types have been mutated and their new types.
  std::map<FunctionType *, FunctionType *> funcTys;

  /// The structs whose types have been mutated and their new types.
  std::map<StructType *, StructType *> structTys;

  /// The constants that have been seen so far when mutating their types.
  /// The initializers of global variables can be mutually recursive, so we need
  /// to be careful when mutating their types.
  std::set<Constant *> seen;

private:
  /// Replace overloaded intrinsic functions. We only need to replace intrinsics
  /// which may be overloaded with different pointer types. Currently, this
  /// only replaces the intrinsics corresponding to libc's memcpy, memmove and
  /// memset functions. This list may need to be expanded.
  bool fixIntrinsics(Module &m) {
    std::map<Function *, Function *> replace;
    LLVMContext &ctxt = m.getContext();
    for (Function &f : m) {
      if (f.isIntrinsic()) {
        FunctionType *fty = f.getFunctionType();
        std::vector<Type *> params;
        Intrinsic::ID id = f.getIntrinsicID();
        switch (id) {
        case Intrinsic::memcpy:
        case Intrinsic::memcpy_inline:
        case Intrinsic::memmove:
          params.push_back(fty->getParamType(0));
          params.push_back(fty->getParamType(1));
          params.push_back(fty->getParamType(2));
          break;
        case Intrinsic::memset:
        case Intrinsic::memset_inline:
          params.push_back(fty->getParamType(0));
          params.push_back(fty->getParamType(2));
          break;
        default:
          // TODO: We may need to support other intrinsics.
          break;
        }
        if (params.size()) {
          FunctionType *type = Intrinsic::getType(ctxt, id, params);
          std::string name = Intrinsic::getName(id, params, &m, type);
          if (f.getName() != name)
            // FIXME: This is deprecated. Use getOrInsertDeclaration instead.
            replace[&f] = Intrinsic::getDeclaration(&m, id, params);
        }
      }
    }

    for (auto &[oldf, newf] : replace)
      oldf->replaceAllUsesWith(newf);
    return replace.size();
  }

  /// Remove address space casts where the source and destination address spaces
  /// are the same.
  void cleanupRedundantCasts(Function &f) {
    // We may have to iterate over the function until convergence. Really should
    // come up with an example of this, if for no other reason than to convince
    // myself that this can happen
    do {
      std::vector<AddrSpaceCastInst *> redundant;
      for (inst_iterator i = inst_begin(f); i != inst_end(f); ++i)
        if (auto *cst = dyn_cast<AddrSpaceCastInst>(&*i))
          if (cst->getSrcAddressSpace() == cst->getDestAddressSpace())
            redundant.push_back(cst);

      if (redundant.empty())
        break;

      for (AddrSpaceCastInst *cst : redundant) {
        cst->replaceAllUsesWith(cst->getPointerOperand());
        cst->eraseFromParent();
      }
    } while (true);
  }

  /// If the given pointer type is in an address space that must be replaced,
  /// return a new pointer type in the default address space.
  PointerType *mutatedType(PointerType *pty) {
    if (pty->getAddressSpace() == KITSUNE_ADDRSPACE)
      return PointerType::getUnqual(pty->getContext());
    return nullptr;
  }

  /// If any parameter of the given function type or its return type must be
  /// mutated, return a new function type, otherwise return nullptr.
  FunctionType *mutatedType(FunctionType *fty) {
    if (funcTys.find(fty) == funcTys.end()) {
      std::vector<Type *> paramTys;
      for (Type *paramTy : fty->params())
        if (Type *newTy = mutatedType(paramTy))
          paramTys.push_back(newTy);
        else
          paramTys.push_back(paramTy);

      Type *retTy = fty->getReturnType();
      if (Type *newTy = mutatedType(fty->getReturnType()))
        retTy = newTy;

      FunctionType *newTy = FunctionType::get(retTy, paramTys, fty->isVarArg());

      funcTys[fty] = nullptr;
      if (newTy != fty)
        funcTys[fty] = newTy;
    }
    return funcTys.at(fty);
  }

  /// If the element of the given array type should be mutated, return a new
  /// array type with the same number of elements and the mutated type.
  /// Otherwise, return nullptr.
  ArrayType *mutatedType(ArrayType *aty) {
    if (Type *newTy = mutatedType(aty->getElementType()))
      return ArrayType::get(newTy, aty->getNumElements());
    return nullptr;
  }

  /// If one or more fields of the struct must be mutated, return a new struct
  /// with the those fields mutated, otherwise return nullptr. If the a new
  /// struct is created it is intended to have the same name as the original
  /// (if the struct is not a literal), but LLVM's uniquing mechanism may well
  /// create a struct with a suffix appended to it.
  StructType *mutatedType(StructType *sty) {
    if (structTys.find(sty) == structTys.end()) {
      std::vector<Type *> elemTys;
      for (Type *elemTy : sty->elements())
        if (Type *newTy = mutatedType(elemTy))
          elemTys.push_back(newTy);
        else
          elemTys.push_back(elemTy);

      LLVMContext &ctxt = sty->getContext();
      bool packed = sty->isPacked();
      if (sty->elements().equals(elemTys))
        structTys[sty] = nullptr;
      else if (sty->isLiteral())
        structTys[sty] = StructType::get(ctxt, elemTys, packed);
      else
        structTys[sty] =
            StructType::create(ctxt, elemTys, sty->getName(), packed);
    }
    return structTys.at(sty);
  }

  /// If the element type of the given vector type should be mutated, return a
  /// new vector with the same number of elements and the mutated type.
  /// Otherwise, return nullptr.
  VectorType *mutatedType(VectorType *vty) {
    if (Type *newTy = mutatedType(vty->getElementType()))
      return VectorType::get(newTy, vty->getElementCount());
    return nullptr;
  }

  /// If the type should be mutated, return the mutated type, otherwise return
  /// nullptr.
  Type *mutatedType(Type *type) {
    if (type->isIntegerTy() or type->isFloatingPointTy() or type->isVoidTy() or
        type->isTokenTy() or type->isLabelTy() or type->isMetadataTy() or
        type->isTargetExtTy() or type->isX86_MMXTy() or type->isX86_AMXTy())
      return nullptr;
    else if (auto *ptrTy = dyn_cast<PointerType>(type))
      return mutatedType(ptrTy);
    else if (auto *funcTy = dyn_cast<FunctionType>(type))
      return mutatedType(funcTy);
    else if (auto *arrTy = dyn_cast<ArrayType>(type))
      return mutatedType(arrTy);
    else if (auto *structTy = dyn_cast<StructType>(type))
      return mutatedType(structTy);
    else if (auto *vectorTy = dyn_cast<VectorType>(type))
      return mutatedType(vectorTy);
    else{
      type->print(llvm::errs(), true);
      llvm::errs() << "type: " << type->getTypeID() << "\n";
      llvm_unreachable(
          "NOT IMPLEMENTED: StripKitsuneAddressSpaces::mutatedType for type");}
    return nullptr;
  }

  bool mutate(Constant &c) {
    bool changed = false;
    if (seen.find(&c) == seen.end()) {
      seen.insert(&c);
      if (Type *newTy = mutatedType(c.getType())) {
        c.mutateType(newTy);
        changed = true;
      }
      for (Value *op : c.operand_values())
        changed |= mutate(*cast<Constant>(op));
    }
    return changed;
  }

  bool mutate(Function &f) {
    if (Type *newTy = mutatedType(f.getFunctionType())) {
      f.mutateValueType(newTy);
      f.mutateType(newTy->getPointerTo());
      return true;
    }
    return false;
  }

  bool mutate(GlobalVariable &g) {
    if (Type *newType = mutatedType(g.getValueType())) {
      // FIXME: Instead of mutating the type here, we should create a new
      // global variable with the correct type and replace the original with
      // the new one. But we currently do not support doing this for functions
      // so we take this shortcut. Eventually, we should fix both globals and
      // functions to do things the right way.
      g.mutateValueType(newType);
      g.mutateType(newType->getPointerTo());

      if (g.hasInitializer())
        mutate(*g.getInitializer());
      return true;
    }
    return false;
  }

  bool mutate(GlobalAlias &a) {
    if (mutatedType(a.getValueType())) {
      llvm_unreachable(
          "NOT IMPLEMENTED: StripKitsuneAddressSpaces::mutate(GlobalAlias)");
    }
    return false;
  }

  bool mutate(GlobalIFunc &f) {
    if (mutatedType(f.getValueType())) {
      llvm_unreachable(
          "NOT IMPLEMENTED: StripKitsuneAddressSpaces::mutate(GlobalIFunc)");
    }
    return false;
  }

  bool mutateGlobalVariables(Module &m) {
    bool changed = false;
    for (GlobalVariable &g : m.globals())
      changed |= mutate(g);
    return changed;
  }

  bool mutateGlobalAliases(Module &m) {
    bool changed = false;
    for (GlobalAlias &a : m.aliases())
      changed |= mutate(a);
    return changed;
  }

  bool mutateIFuncs(Module &m) {
    bool changed = false;
    for (GlobalIFunc &f : m.ifuncs())
      changed |= mutate(f);
    return changed;
  }

  bool mutateFunctions(Module &m) {
    bool changed = false;
    for (Function &f : m)
      changed |= mutate(f);

    for (Function &f : m) {
      for (Argument &arg : f.args())
        if (Type *newType = mutatedType(arg.getType()))
          arg.mutateType(newType);

      for (inst_iterator i = inst_begin(f); i != inst_end(f); ++i) {
        // Some instructions need special handling.
        if (auto *call = dyn_cast<CallInst>(&*i))
          if (Type *newTy = mutatedType(call->getFunctionType()))
            call->mutateFunctionType(cast<FunctionType>(newTy));

        // I am not sure if we strictly need this. But if the instruction is
        // never used, I can imagine nothing will change the type which is
        // obviously not what we want.
        if (Type *newTy = mutatedType(i->getType()))
          i->mutateType(newTy);

        // The values should not need any special tratment.
        for (Value *op : i->operand_values())
          if (Type *newTy = mutatedType(op->getType()))
            op->mutateType(newTy);
      }

      // We may be left with addrspacecasts where the source and destination
      // address spaces are the same. These are not allowed and will cause the
      // verifier to fail, so clean those up now.
      cleanupRedundantCasts(f);
    }

    // Some LLVM intrinsics have different names depending on any pointers
    // passed to them. If the types of the pointer arguments to these have
    // changed, the functions need to be fixed.
    changed |= fixIntrinsics(m);

    return changed;
  }

public:
  StripKitsuneAddrSpaceImpl() = default;

  bool run(Module &m) {
    bool changed = false;

    changed |= mutateGlobalAliases(m);
    changed |= mutateGlobalVariables(m);
    changed |= mutateIFuncs(m);
    changed |= mutateFunctions(m);

    return changed;
  }
};

PreservedAnalyses StripKitsuneAddrSpacePass::run(Module &m,
                                                 ModuleAnalysisManager &mam) {
  if (StripKitsuneAddrSpaceImpl().run(m))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
