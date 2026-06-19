//===- AddrSpace.cpp - Support for Kitsune's address spaces ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities for Kitsune's address spaces.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/AddrSpace.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

using namespace llvm;

namespace {

class StripAddrSpaces {
private:
  /// The functions whose types have been mutated and their new types.
  SmallDenseMap<FunctionType *, FunctionType *> funcTys;

  /// The structs whose types have been mutated and their new types.
  SmallDenseMap<StructType *, StructType *> structTys;

  /// The constants that have been seen so far when mutating their types. The
  /// initializers of global variables can be mutually recursive, so we must be
  /// careful when mutating their types.
  SmallSet<Constant *, 8> seen;

private:
  bool fixIntrinsic(Function &f, ArrayRef<unsigned> paramIndices);
  bool fixIntrinsic(Function &f);
  bool fixIntrinsics(Module &m);

  bool cleanupRedundantCasts(Function &f);

  PointerType *mutatedType(PointerType *pty);
  FunctionType *mutatedType(FunctionType *fty);
  ArrayType *mutatedType(ArrayType *aty);
  StructType *mutatedType(StructType *sty);
  VectorType *mutatedType(VectorType *vty);
  Type *mutatedType(Type *type);

  bool mutate(Constant &c);
  bool mutate(Function &f);
  bool mutate(GlobalVariable &g);
  bool mutate(GlobalAlias &a);
  bool mutate(GlobalIFunc &f);
  bool mutateGlobalVariables(Module &m);
  bool mutateGlobalAliases(Module &m);
  bool mutateIFuncs(Module &m);
  bool mutateFunctions(Module &m);

public:
  bool run(Module &m);
};

} // namespace

bool StripAddrSpaces::fixIntrinsic(Function &f,
                                   ArrayRef<unsigned> paramIndices) {
  LLVMContext &ctx = f.getContext();
  Module *m = f.getParent();
  Intrinsic::ID id = f.getIntrinsicID();
  FunctionType *fty = f.getFunctionType();

  SmallVector<Type *, 4> params;
  for (unsigned i : paramIndices)
    params.push_back(fty->getParamType(i));

  FunctionType *expectedTy = Intrinsic::getType(ctx, id, params);
  std::string expectedName = Intrinsic::getName(id, params, m, expectedTy);
  if (f.getName() != expectedName) {
    Function *newf = Intrinsic::getOrInsertDeclaration(m, id, params);
    f.replaceAllUsesWith(newf);
    f.eraseFromParent();
    return true;
  }
  return false;
}

bool StripAddrSpaces::fixIntrinsic(Function &f) {
  // It is not clear why these intrinsics have to be handled specially, but
  // simply calculating the intrinsic name with the function parameter types
  // does not work correctly.
  switch (f.getIntrinsicID()) {
  case Intrinsic::memcpy:
  case Intrinsic::memcpy_inline:
  case Intrinsic::memmove:
    return fixIntrinsic(f, {0, 1, 2});
  case Intrinsic::memset:
  case Intrinsic::memset_inline:
    return fixIntrinsic(f, {0, 2});
  default:
    return false;
  }
}

/// Replace overloaded intrinsic functions. We only need to replace intrinsics
/// which may be overloaded with different pointer types. Currently, this only
/// replaces the intrinsics corresponding to libc's memcpy, memmove and memset
/// functions. This list may need to be expanded.
bool StripAddrSpaces::fixIntrinsics(Module &m) {
  SmallVector<Function *, 8> fns;
  for (Function &f : m)
    fns.push_back(&f);

  bool changed = false;
  for (Function *f : fns)
    changed |= fixIntrinsic(*f);
  return changed;
}

/// Remove address space casts where the source and destination address spaces
/// are the same.
bool StripAddrSpaces::cleanupRedundantCasts(Function &f) {
  bool changed = false;

  // We iterate over the function until convergence. I should come up with an
  // example of this, if for no other reason than to convince myself that more
  // than one pass over the function may be necessary.
  do {
    SmallVector<AddrSpaceCastInst *, 4> redundant;
    for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i)
      if (auto *cst = dyn_cast<AddrSpaceCastInst>(&*i))
        if (cst->getSrcAddressSpace() == cst->getDestAddressSpace())
          redundant.push_back(cst);

    if (redundant.empty())
      break;

    for (AddrSpaceCastInst *cst : redundant) {
      cst->replaceAllUsesWith(cst->getPointerOperand());
      cst->eraseFromParent();
    }
    changed |= true;
  } while (true);

  return changed;
}

/// If the given pointer type is in an address space that must be replaced,
/// return a new pointer type in the default address space.
PointerType *StripAddrSpaces::mutatedType(PointerType *pty) {
  if (pty->getAddressSpace() == KitAS::Mobile)
    return PointerType::getUnqual(pty->getContext());
  return nullptr;
}

/// If any parameter of the given function type or its return type must be
/// mutated, return a new function type, otherwise return nullptr.
FunctionType *StripAddrSpaces::mutatedType(FunctionType *fty) {
  if (funcTys.find(fty) == funcTys.end()) {
    SmallVector<Type *, 4> paramTys;
    for (Type *paramTy : fty->params())
      if (Type *newTy = mutatedType(paramTy))
        paramTys.push_back(newTy);
      else
        paramTys.push_back(paramTy);

    Type *retTy = fty->getReturnType();
    if (Type *newTy = mutatedType(fty->getReturnType()))
      retTy = newTy;

    FunctionType *newTy = FunctionType::get(retTy, paramTys, fty->isVarArg());
    if (newTy != fty)
      funcTys[fty] = newTy;
    else
      funcTys[fty] = nullptr;
  }
  return funcTys.at(fty);
}

/// If the element of the given array type should be mutated, return a new array
/// type with the same number of elements and the mutated type. Otherwise,
/// return nullptr.
ArrayType *StripAddrSpaces::mutatedType(ArrayType *aty) {
  if (Type *newTy = mutatedType(aty->getElementType()))
    return ArrayType::get(newTy, aty->getNumElements());
  return nullptr;
}

/// If one or more fields of the struct must be mutated, return a new struct
/// with the those fields mutated, otherwise return nullptr. If the a new struct
/// is created it is intended to have the same name as the original (if the
/// struct is not a literal), but LLVM's uniquing mechanism may well create a
/// struct with a suffix appended to it.
StructType *StripAddrSpaces::mutatedType(StructType *sty) {
  if (structTys.find(sty) == structTys.end()) {
    SmallVector<Type *, 4> elemTys;
    for (Type *elemTy : sty->elements())
      if (Type *newTy = mutatedType(elemTy))
        elemTys.push_back(newTy);
      else
        elemTys.push_back(elemTy);

    LLVMContext &ctx = sty->getContext();
    bool pack = sty->isPacked();
    if (sty->elements().equals(elemTys))
      structTys[sty] = nullptr;
    else if (sty->isLiteral())
      structTys[sty] = StructType::get(ctx, elemTys, pack);
    else
      structTys[sty] = StructType::create(ctx, elemTys, sty->getName(), pack);

    // The name of the new struct will not be exactly the same as the old
    // struct. Swap the names to ensure that the new struct has the same name
    // after all the old instances are replaced.
    if (structTys.at(sty) && !sty->isLiteral()) {
      StructType *newSty = structTys.at(sty);
      std::string t = sty->getName().str();
      sty->setName(newSty->getName());
      newSty->setName(t);
    }
  }
  return structTys.at(sty);
}

/// If the element type of the given vector type should be mutated, return a new
/// vector with the same number of elements and the mutated type. Otherwise,
/// return nullptr.
VectorType *StripAddrSpaces::mutatedType(VectorType *vty) {
  if (Type *newTy = mutatedType(vty->getElementType()))
    return VectorType::get(newTy, vty->getElementCount());
  return nullptr;
}

/// If the type should be mutated, return the mutated type, otherwise return
/// nullptr.
Type *StripAddrSpaces::mutatedType(Type *type) {
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
  llvm_unreachable("NOT IMPLEMENTED: StripAddressSpaces::mutatedType for type");
}

bool StripAddrSpaces::mutate(Constant &c) {
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

bool StripAddrSpaces::mutate(Function &f) {
  if (Type *newTy = mutatedType(f.getFunctionType())) {
    f.mutateValueType(newTy);
    f.mutateType(PointerType::getUnqual(f.getContext()));
    return true;
  }
  return false;
}

bool StripAddrSpaces::mutate(GlobalVariable &g) {
  if (Type *newType = mutatedType(g.getValueType())) {
    // FIXME: Instead of mutating the type here, we should create a new global
    // variable with the correct type and replace the original with the new one.
    // But we currently do not support doing this for functions, so we take this
    // shortcut. Eventually, we should fix both globals and functions to do
    // things the right way.
    g.mutateValueType(newType);
    g.mutateType(PointerType::getUnqual(g.getContext()));

    if (g.hasInitializer())
      mutate(*g.getInitializer());
    return true;
  }
  return false;
}

bool StripAddrSpaces::mutate(GlobalAlias &a) {
  if (mutatedType(a.getValueType())) {
    llvm_unreachable(
        "NOT IMPLEMENTED: StripAddressSpaces::mutate(GlobalAlias)");
  }
  return false;
}

bool StripAddrSpaces::mutate(GlobalIFunc &f) {
  if (mutatedType(f.getValueType())) {
    llvm_unreachable(
        "NOT IMPLEMENTED: StripAddressSpaces::mutate(GlobalIFunc)");
  }
  return false;
}

bool StripAddrSpaces::mutateGlobalVariables(Module &m) {
  bool changed = false;
  for (GlobalVariable &g : m.globals())
    changed |= mutate(g);
  return changed;
}

bool StripAddrSpaces::mutateGlobalAliases(Module &m) {
  bool changed = false;
  for (GlobalAlias &a : m.aliases())
    changed |= mutate(a);
  return changed;
}

bool StripAddrSpaces::mutateIFuncs(Module &m) {
  bool changed = false;
  for (GlobalIFunc &f : m.ifuncs())
    changed |= mutate(f);
  return changed;
}

bool StripAddrSpaces::mutateFunctions(Module &m) {
  bool changed = false;

  for (Function &f : m)
    changed |= mutate(f);

  for (Function &f : m) {
    for (Argument &arg : f.args())
      if (Type *newType = mutatedType(arg.getType()))
        arg.mutateType(newType);

    for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i) {
      // Some instructions need special handling.
      if (auto *call = dyn_cast<CallBase>(&*i))
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
    changed |= cleanupRedundantCasts(f);
  }

  // Some LLVM intrinsics have different names depending on the pointers
  // passed to them. If the types of the pointer arguments to these have
  // changed, the functions need to be fixed.
  changed |= fixIntrinsics(m);

  return changed;
}

bool StripAddrSpaces::run(Module &m) {
  bool changed = false;

  changed |= mutateGlobalAliases(m);
  changed |= mutateGlobalVariables(m);
  changed |= mutateIFuncs(m);
  changed |= mutateFunctions(m);

  return changed;
}

bool llvm::stripKitAddrSpaces(Module &m) { return StripAddrSpaces().run(m); }
