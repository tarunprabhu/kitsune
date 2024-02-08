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
#include "llvm/Support/CommandLine.h"

#include <map>
#include <set>

using namespace llvm;

// This is only really useful for testing.
static cl::opt<bool> clDisableStripKitsuneAddrspace(
    "disable-strip-kitsune-addrspaces", cl::init(false), cl::Hidden,
    cl::desc("Do not strip kitsune address spaces"));

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
  /// The kitsune memory allocation instrinsics and kitrt runtime alloction
  /// function declarations must retain the Kitsune address space. Since these
  /// have been removed, put them back. The returned value must be cast to strip
  /// away the address space.
  bool fixKitsuneAllocFuncs(Module &m) {
    LLVMContext &ctxt = m.getContext();
    Type *defPtr = PointerType::getUnqual(ctxt);
    Type *asPtr = PointerType::get(ctxt, KITSUNE_ADDRSPACE);
    std::map<CallBase *, Function *> calls;

    for (Function &f : m) {
      if (not f.size()) {
        // TODO: At some point we will have registered the function names with
        // TLI, so we should look up them there.
        if (f.getIntrinsicID() == Intrinsic::kitsune_mobile_alloc or
            f.getName() == "__kitrt_default_mem_alloc" or
            f.getName() == "__kitcuda_mem_alloc_managed" or
            f.getName() == "__kithip_mem_alloc_managed") {
          // Find all the uses before mutating the type. If we change the order,
          // getCalledFunction() will always return nullptr since the type of
          // the callee and that of the call will not match.
          for (Use &u : f.uses()) {
            if (auto *call = dyn_cast<CallBase>(u.getUser())) {
              if (call->getCalledFunction() == &f)
                calls[call] = &f;
              else
                llvm_unreachable(
                    "fixKitsuneAlloc: Unexpected use of function in call");
            } else {
              llvm_unreachable("fixKitsuneAlloc: Unexpected use of function");
            }
          }
          ArrayRef<Type *> params = f.getFunctionType()->params();
          FunctionType *fty = FunctionType::get(asPtr, params, f.isVarArg());
          f.mutateValueType(fty);
          f.mutateType(PointerType::getUnqual(ctxt));
        }
      }
    }

    for (auto &[call, f] : calls) {
      call->mutateFunctionType(f->getFunctionType());
      call->mutateType(asPtr);

      // Collect the uses of the call before creating the cast instruction so
      // the latter does not get included in the use list.
      std::vector<Use *> uses;
      for (Use &u : call->uses())
        uses.push_back(&u);

      // Cast the result of the call to the right address space, but don't add
      // the cast instruction into the function just yet. It is probably fine to
      // do this now, but let's hold off anyway.
      auto *cst = CastInst::CreatePointerBitCastOrAddrSpaceCast(call, defPtr);
      for (Use *u : uses)
        u->getUser()->setOperand(u->getOperandNo(), cst);

      // All the uses have been replaced. Now add the cast immediately after the
      // call and fix the types of the call.
      cst->insertAfter(call);
    }
    return calls.size();
  }

  /// The kitsune memory free instrinsics and kitrt runtime memory free
  /// function declarations must retain the Kitsune address space. Since these
  /// have been removed, put them back. These functions take a single pointer
  /// argument which must be cast to the correct address space after the
  /// declarations have been fixed.
  bool fixKitsuneFreeFuncs(Module &m) {
    LLVMContext &ctxt = m.getContext();
    Type *asPtr = PointerType::get(ctxt, KITSUNE_ADDRSPACE);
    std::map<CallBase *, Function *> calls;
    for (Function &f : m) {
      if (not f.size()) {
        // TODO: At some point we will have registered the function names with
        // TLI, so we should look up them there.
        if (f.getIntrinsicID() == Intrinsic::kitsune_mobile_free or
            f.getName() == "__kitrt_default_mem_free" or
            f.getName() == "__kitcuda_mem_free" or
            f.getName() == "__kithip_mem_free") {
          // Find all the uses before mutating the type. If we change the order,
          // getCalledFunction() will always return nullptr since the type of
          // the callee and that of the call will not match.
          for (Use &u : f.uses()) {
            if (auto *call = dyn_cast<CallBase>(u.getUser())) {
              if (call->getCalledFunction() == &f)
                calls[call] = &f;
              else
                llvm_unreachable(
                    "fixKitsuneFree: Unexpected use of function in call");
            } else {
              llvm_unreachable("fixKitsuneFree: Unexpected use of function");
            }
          }

          // Now we can mutate the function type and those of its arguments.
          Type *params[] = {asPtr};
          Type *ret = f.getReturnType();
          FunctionType *fty = FunctionType::get(ret, params, f.isVarArg());
          f.mutateValueType(fty);
          f.mutateType(PointerType::getUnqual(ctxt));
          f.getArg(0)->mutateType(asPtr);
        }
      }
    }

    for (auto &[call, f] : calls) {
      Value *arg = call->getArgOperand(0);
      CastInst *cst =
          CastInst::CreatePointerBitCastOrAddrSpaceCast(arg, asPtr, "");
      cst->insertBefore(call->getIterator());
      call->setArgOperand(0, cst);
      call->mutateFunctionType(f->getFunctionType());
    }

    return calls.size();
  }

  /// Put the address spaces back in the Kitsune memory allocation/free
  /// functions and adjust the call sites.
  bool fixKitsuneFuncs(Module &m) {
    bool changed = false;
    changed |= fixKitsuneAllocFuncs(m);
    changed |= fixKitsuneFreeFuncs(m);
    return changed;
  }

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
            replace[&f] = Intrinsic::getOrInsertDeclaration(&m, id, params);
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
        type->isTargetExtTy() or type->isX86_AMXTy())
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
    else
      llvm_unreachable(
          "NOT IMPLEMENTED: StripKitsuneAddressSpaces::mutatedType for type");
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
      f.mutateType(PointerType::getUnqual(f.getContext()));
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
      g.mutateType(PointerType::getUnqual(g.getContext()));

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

    // This will have removed the address spaces from the Kitsune memory
    // allocation and free functions/intrinsics which is very wrong. It is
    // probably easier to put them back than to stop them from being removed in
    // the first place, so do that now.
    changed |= fixKitsuneFuncs(m);

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
  if (clDisableStripKitsuneAddrspace)
    return PreservedAnalyses::all();

  if (StripKitsuneAddrSpaceImpl().run(m))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
