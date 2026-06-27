//===- Verifier.cpp - Interface for Kitsune-specific verifiers ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Kitsune-specific verifiers
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/Verifier.h"
#include "ArgAttrsImpl.h"
#include "AttrsImpl.h"
#include "FuncAttrsImpl.h"
#include "GVAttrsImpl.h"
#include "InstAttrsImpl.h"
#include "LoopAttrsImpl.h"
#include "ModuleAttrsImpl.h"
#include "VerifierImpl.h"
#include "kitsune/Core/Attrs.h"
#include "kitsune/Core/IntrinsicUtils.h"
#include "kitsune/Core/TTUtils.h"
#include "kitsune/Core/ValueUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

using namespace llvm;

KitVerifier &KitVerifier::verify(const Argument &a) {
  for (const MDNode &attr : detail::attrs(a))
    detail::verifyAttr(*this, a, detail::getRawAttrName(attr));
  return *this;
}

KitVerifier &KitVerifier::verify(const Function &f) {
  for (const MDNode &attr : detail::attrs(f))
    detail::verifyAttr(*this, f, detail::getRawAttrName(attr));

  for (const Argument &a : f.args())
    verify(a);

  // Functions without a body may have attributes that must be verified. The
  // same for the arguments. But don't bother with anything else.
  if (!f.size())
    return *this;

  for (const_inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i)
    verify(*i);

  DominatorTree dt(const_cast<Function &>(f));
  LoopInfo li(dt);
  for (const Loop *loop : li)
    for (const MDNode &attr : detail::attrs(*loop))
      detail::verifyAttr(*this, *loop, detail::getRawAttrName(attr));

  return *this;
}

KitVerifier &KitVerifier::verify(const GlobalAlias &g) {
  // Nothing Kitsune-specific to be done here.
  return *this;
}

KitVerifier &KitVerifier::verify(const GlobalIFunc &g) {
  // Nothing Kitsune-specific to be done here.
  return *this;
}

KitVerifier &KitVerifier::verify(const GlobalVariable &g) {
  for (const MDNode &attr : detail::attrs(g))
    detail::verifyAttr(*this, g, detail::getRawAttrName(attr));
  return *this;
}

KitVerifier &KitVerifier::verifyIntrMobileInit(const CallBase &call) {
  auto isSupportedScalar = [](Type *ty) {
    return ty->isIntegerTy(1) || ty->isIntegerTy(8) || ty->isIntegerTy(16) ||
           ty->isIntegerTy(32) || ty->isIntegerTy(64) || ty->isFloatTy() ||
           ty->isDoubleTy();
  };

  LLVMContext &ctx = call.getContext();
  Type *i32 = Type::getInt32Ty(ctx);
  Type *initTy = call.getArgOperand(3)->getType();
  unsigned numArgs = call.arg_size();
  switch (numArgs) {
  case 4:
    if (initTy->isPointerTy())
      check(false, DiagID::ErrMobileInitExpectSize, getName(call), *i32);
    else
      check(isSupportedScalar(initTy), DiagID::ErrMobileInitUnsupportedType,
            *initTy, call);
    break;
  case 5:
    check(initTy->isPointerTy(), DiagID::ErrMobileInitExpectPointer,
          getName(call));
    check(isInt32(call.getArgOperand(4)), DiagID::ErrMobileInitExpectSize,
          getName(call), *i32);
    break;
  default:
    check(false, DiagID::ErrNumCallArgs, call, initTy->isPointerTy() ? 5 : 4,
          numArgs);
    break;
  }
  return *this;
}

KitVerifier &KitVerifier::verifyIntrReduce(const CallBase &call, Value *unitVal,
                                           Value *reducerVal,
                                           unsigned extraArgNum) {
  // This is a best effort attempt at verifying the call. In most cases, the
  // reducer argument to the call will be a function. It is possible, though
  // unlikely that this will be some other value.
  Function *reducer = dyn_cast<Function>(reducerVal);
  if (!reducer)
    return *this;

  LLVMContext &ctx = call.getContext();
  Type *ptr = PointerType::getUnqual(ctx);

  FunctionType *reducerTy = reducer->getFunctionType();
  unsigned numParams = reducerTy->getNumParams();
  StringRef reducerName = reducer->getName();

  check(reducerTy->getReturnType()->isVoidTy(), DiagID::ErrReducerReturnType,
        reducerName);
  if (numParams < 2) {
    check(false, DiagID::ErrReducerMinArgs, reducerName);
    return *this;
  }

  Argument *destArg = reducer->getArg(0);
  check(destArg->getType()->isPointerTy(), DiagID::ErrReducerTypeMismatch,
        reducerName, 1, *ptr);

  Type *unitTy = unitVal->getType();
  Argument *valArg = reducer->getArg(1);
  check(valArg->getType() == unitVal->getType(), DiagID::ErrReducerTypeMismatch,
        reducerName, 2, *unitTy);

  unsigned numArgs = call.arg_size();
  bool hasExtraArgs = numArgs > extraArgNum;
  if (hasExtraArgs) {
    unsigned numExtraFunc = numParams - 2;
    unsigned numExtraCall = numArgs - extraArgNum;
    llvm::outs() << numExtraFunc << " " << numExtraCall << "\n";
    if (numExtraCall != numExtraFunc) {
      check(false, DiagID::ErrReducerNumParams, reducerName, numExtraCall + 2,
            numParams);
    } else {
      for (unsigned i = 2, j = extraArgNum; i < numParams; ++i, ++j) {
        Type *paramTy = reducerTy->getParamType(i);
        Type *argTy = call.getArgOperand(j)->getType();
        check(paramTy == argTy, DiagID::ErrReducerTypeMismatch, reducerName,
              i + 1, *argTy);
      }
    }
  } else {
    check(numParams == 2, DiagID::ErrReducerNumParams, reducerName, 2,
          numParams);
  }

  return *this;
}

KitVerifier &KitVerifier::verifyIntrReduce0(const CallBase &call) {
  Value *val = call.getArgOperand(3);
  Value *unit = call.getArgOperand(4);
  Value *reducer = call.getArgOperand(5);

  Type *valTy = val->getType();
  Type *unitTy = unit->getType();

  check(valTy == unitTy, DiagID::ErrReduceUnitValMismatch);

  return verifyIntrReduce(call, unit, reducer, 6);
}

KitVerifier &KitVerifier::verifyIntrReduce1(const CallBase &call) {
  Value *unit = call.getArgOperand(5);
  Value *reducer = call.getArgOperand(6);

  return verifyIntrReduce(call, unit, reducer, 7);
}

KitVerifier &KitVerifier::verify(const CallBase &call) {
  Intrinsic::ID id = call.getIntrinsicID();
  if (isKitIntrinsic(id)) {
    std::optional<TTID> tt = getTTIDFromKitIntrCall(call);
    if (!tt.has_value())
      check(false, DiagID::ErrKitIntrNoTTID);
    else if (isKitIntrinsicCPU(id))
      check(isCPUTT(*tt), DiagID::ErrKitIntrTTIDNotCPU);
    else if (isKitIntrinsicGPU(id))
      check(isGPUTT(*tt), DiagID::ErrKitIntrTTIDNotGPU);
  }

  switch (call.getIntrinsicID()) {
  case Intrinsic::kit_mobile_init:
    return verifyIntrMobileInit(call);
  case Intrinsic::kit_reduce_0:
    return verifyIntrReduce0(call);
  case Intrinsic::kit_reduce_1:
    return verifyIntrReduce1(call);
  case Intrinsic::kit_runtime_set_xnack:
  case Intrinsic::kit_runtime_set_y_axis_kernel_launch:
    if (std::optional<TTID> tt = getTTIDFromKitIntrCall(call))
      check(*tt == TTID::Hip, DiagID::ErrKitIntrWrongTTID, TTID::Hip);
    return *this;
  default:
    return *this;
  }
}

KitVerifier &KitVerifier::verify(const Instruction &inst) {
  for (const MDNode &attr : detail::attrs(inst))
    detail::verifyAttr(*this, inst, detail::getRawAttrName(attr));

  if (const auto *call = dyn_cast<CallBase>(&inst))
    return verify(*call);
  return *this;
}

KitVerifier &KitVerifier::verify(const Module &m) {
  for (const Function &f : m.functions())
    verify(f);

  for (const GlobalAlias &g : m.aliases())
    verify(g);

  for (const GlobalIFunc &ifunc : m.ifuncs())
    verify(ifunc);

  for (const GlobalVariable &g : m.globals())
    verify(g);

  for (const MDNode &attr : detail::attrs(m))
    detail::verifyAttr(*this, m, detail::getRawAttrName(attr));

  // Some checks of "related" attributes cannot be reasonably added to the
  // verifier of either attribute. Do those here.

  // There can be at most one global variable containing device code per
  // tapir target.
  SmallDenseMap<TTID, unsigned> dcGlobals(4);
  for (const GlobalVariable &g : m.globals())
    if (std::optional<TTID> tt = getDeviceCodeAttr(g))
      ++dcGlobals[*tt];

  for (const auto &[tt, n] : dcGlobals)
    check(n <= 1, DiagID::ErrTooManyDeviceCodeGlobals, tt);

  // If a global variable containing bitcode exists, then a corresponding global
  // containing device code must also exist. The reverse is not true. Once the
  // device code has been generated, the global containing bitcode is removed.
  SmallDenseMap<TTID, unsigned> bcGlobals(4);
  for (const GlobalVariable &g : m.globals())
    if (std::optional<TTID> tt = getBitCodeAttr(g))
      ++bcGlobals[*tt];

  for (const auto &[tt, n] : bcGlobals) {
    check(n <= 1, DiagID::ErrTooManyBitCodeGlobals, tt);
    check(dcGlobals.contains(tt), DiagID::ErrMissingDeviceCodeGlobal, tt);
  }

  return *this;
}

bool llvm::verifyFunction(const Function &f, bool kitOnly, raw_ostream *os) {
  if (kitOnly)
    return KitVerifier(os).verify(f).result();

  // LLVM's verifyFunction will call this function with kitOnly == true, so
  // Kitsune-specific verification will be performed then. But the value
  // returned by llvm::verifyFunction will be the opposite of what this function
  // should return.
  return !verifyFunction(f, os);
}

bool llvm::verifyModule(const Module &m, bool kitOnly, raw_ostream *os) {
  if (kitOnly)
    return KitVerifier(os).verify(m).result();

  // LLVM's verifyModule will call this function with kitOnly == true, so
  // Kitsune-specific verification will be performed then. But the value
  // returned by llvm::verifyModule will be the opposite of what this function
  // should return.
  return !verifyModule(m, os);
}
