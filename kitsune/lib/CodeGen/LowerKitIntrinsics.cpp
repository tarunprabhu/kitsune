//===- LowerKitIntrinsics.cpp - Lower Kitsune-specific intrinsics ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower kitsune-specific intrinsics.
//
// These are nearly always lowered to a corresponding Kitsune's runtime
// function, but this need not always be the case.
//
//===----------------------------------------------------------------------===//

#include "kitsune/CodeGen/LowerKitIntrinsics.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/InstUtils.h"
#include "kitsune/Core/IntrinsicUtils.h"
#include "kitsune/Core/LibFuncs.h"
#include "kitsune/Core/ValueUtils.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/InitializePasses.h"

#include <map>

#define DEBUG_TYPE "kit-lower-intrinsics"

using namespace llvm;

using LibFuncMap = std::map<TTID, KitFunc>;
using KitIntrLibFuncMap = std::map<Intrinsic::ID, LibFuncMap>;
#define GET_INTR_LIBFUNC_MAP
#include "kitsune/Core/IntrLibFuncMap.inc"
static const KitIntrLibFuncMap intrLibFuncMap = INTR_LIBFUNC_MAP;

static bool requiresCustomLowering(const CallInst &call) {
  switch (call.getIntrinsicID()) {
#define GET_INTR_LOWERING_SPEC
#define INTR(NAME, CUSTOM_LOWERING, ALLOW_PARAM_CAST, ALLOW_RETURN_CAST)       \
  case Intrinsic::NAME: return CUSTOM_LOWERING;
#include "kitsune/Core/IntrLibFuncMap.inc"
  }
  llvm_unreachable("requiresCustomLowering: Intrinsic ID not handled");
}

static bool allowParamCast(const CallInst &call) {
  switch (call.getIntrinsicID()) {
#define GET_INTR_LOWERING_SPEC
#define INTR(NAME, CUSTOM_LOWERING, ALLOW_PARAM_CAST, ALLOW_RETURN_CAST)       \
  case Intrinsic::NAME: return ALLOW_PARAM_CAST;
#include "kitsune/Core/IntrLibFuncMap.inc"
  }
  llvm_unreachable("allowParamCast: Intrinsic ID not handled");
}

static bool allowReturnCast(const CallInst &call) {
  switch (call.getIntrinsicID()) {
#define GET_INTR_LOWERING_SPEC
#define INTR(NAME, CUSTOM_LOWERING, ALLOW_PARAM_CAST, ALLOW_RETURN_CAST)       \
  case Intrinsic::NAME: return ALLOW_RETURN_CAST;
#include "kitsune/Core/IntrLibFuncMap.inc"
  }
  llvm_unreachable("allowReturnCast: Intrinsic ID not handled");
}

static FunctionCallee getRuntimeFunc(CallInst &call, KitFunc rtFunc) {
  Module &m = *call.getModule();
  return getOrInsertLibFunc(m, rtFunc);
}

// Get the kitsune runtime function that will replace the intrinsic called in
// the given call instruction.
static FunctionCallee getRuntimeFunc(CallInst &call) {
  Intrinsic::ID id = call.getIntrinsicID();
  const LibFuncMap &libFuncMap = intrLibFuncMap.at(id);

  TTID tt = *getTTIDFromKitIntrCall(call);
  assert(libFuncMap.find(tt) != libFuncMap.end() &&
         "No library function for tapir target");

  return getRuntimeFunc(call, libFuncMap.at(tt));
}

// If the type of \p v does not match \p dstTy, insert a cast using the given
// builder and return the casted value. Otherwise, simply return v.
Value *maybeCast(Value *v, Type *dstTy, IRBuilder<> &builder) {
  Type *srcTy = v->getType();
  if (srcTy == dstTy)
    return v;
  else if (srcTy->isPointerTy() && dstTy->isPointerTy())
    return builder.CreatePointerBitCastOrAddrSpaceCast(v, dstTy);
  else if (srcTy->isIntegerTy(1) && dstTy->isIntegerTy())
    return builder.CreateIntCast(v, dstTy, /*isSigned=*/false);
  else if (srcTy->isIntegerTy() && dstTy->isIntegerTy())
    return builder.CreateIntCast(v, dstTy, /*isSigned=*/true);
  else if (srcTy->isFloatingPointTy() && dstTy->isFloatingPointTy())
    return builder.CreateFPCast(v, dstTy);
  else if (srcTy->isPointerTy() && dstTy->isIntegerTy())
    return builder.CreatePtrToInt(v, dstTy);
  else if (srcTy->isIntegerTy() && srcTy->isPointerTy())
    return builder.CreateIntToPtr(v, dstTy);
  llvm_unreachable("maybeCast: Cast kind not yet implemented");
};

// Get the arguments that must be passed to the runtime function \p rtFunc in a
// default lowering. Since the first argument of the intrinsic will be the TTID,
// that will be skipped. All the other arguments will be returned in order. If
// the type of some argument does not match that of the corresponding parameter
// of \p rtFunc, and casting is permitted, a cast will be inserted using the
// given builder and the casted value will be added to the returned list.
SmallVector<Value *, 4> getDefaultLoweringArgs(CallInst &call,
                                               FunctionCallee rtFunc,
                                               IRBuilder<> &builder) {
  SmallVector<Value *, 4> args;
  FunctionType *funcTy = rtFunc.getFunctionType();
  for (unsigned i = 1; i < call.arg_size(); ++i) {
    Value *arg = call.getArgOperand(i);
    if (allowParamCast(call))
      arg = maybeCast(arg, funcTy->getParamType(i - 1), builder);
    args.push_back(arg);
  }
  return args;
}

// Return a new attribute list which is exactly the same as the given
// attribute list \ref attrs except that the attributes at index \ref src of
// \ref call's attribute list are added to index \ref dst of \ref attrs. The
// newly created attribute list is returned.
static AttributeList addAttrsFrom(AttributeList attrs, unsigned dst,
                                  const CallInst &call, unsigned src) {
  LLVMContext &ctx = call.getContext();
  AttributeList callAttrs = call.getAttributes();
  for (const Attribute &attr : callAttrs.getAttributes(src))
    attrs = attrs.addAttributeAtIndex(ctx, dst, attr);
  return attrs;
}

// Return a new attribute list which is exactly the same as the given attribute
// list \ref attrs except that the attributes at index \ref src of \ref call's
// attribute list are added to index \ref src of \ref attrs. The newly created
// attribute list is returned.
static AttributeList addAttrsFrom(AttributeList attrs, const CallInst &call,
                                  unsigned src) {
  return addAttrsFrom(attrs, src, call, src);
}

// Create a new attribute list that will eventually be applied to the
// replacement of \p call. \p call is expected to be a direct call to a
// Kitsune-specific intrinsic. Since the first argument to such intrinsics will
// always be a TTID, that is skipped. The remaining non-variadic arguments are
// assumed to be passed as-is to the new call, so their attributes are copied.
static AttributeList createNewAttrList(const CallInst &call) {
  AttributeList attrs;
  attrs = addAttrsFrom(attrs, call, AttributeList::FunctionIndex);
  attrs = addAttrsFrom(attrs, call, AttributeList::ReturnIndex);
  for (size_t i = 1; i < getNumNonVariadicArgs(call); ++i) {
    unsigned src = AttributeList::FirstArgIndex + i;
    unsigned dst = AttributeList::FirstArgIndex + i - 1;
    attrs = addAttrsFrom(attrs, dst, call, src);
  }
  return attrs;
}

// Create a new call to the given function to replace an existing call. The
// debug info, metadata, calling convention and tail call kind will be copied
// over from the original call. However, the attributes will not be copied. The
// new call is returned, but the original call will remain unchanged.
//
// The attributes are not copied because there are some intrinsics where the
// attributes cannot be copied over directly. To avoid having conditional
// statements in this function, we require the attributes to be copied over by
// callers.
static Value *createNewCallFor(CallInst &call, FunctionCallee f,
                               ArrayRef<Value *> args, IRBuilder<> &builder) {
  CallInst *newCall = builder.CreateCall(f, args);
  newCall->cloneDebugInfoFrom(&call);
  newCall->copyMetadata(call);
  newCall->setCallingConv(call.getCallingConv());
  newCall->takeName(&call);
  newCall->setAttributes(createNewAttrList(call));

  // Because the result of the lowered intrinsic may be cast to a different
  // type (typically this will be an address space cast), tail calls cannot be
  // guaranteed.
  CallInst::TailCallKind tck = call.getTailCallKind();
  if (tck == CallInst::TCK_MustTail)
    newCall->setTailCallKind(CallInst::TCK_Tail);
  else
    newCall->setTailCallKind(tck);

  Value *newInst = newCall;
  if (allowReturnCast(call))
    newInst = maybeCast(newCall, call.getType(), builder);

  call.replaceAllUsesWith(newInst);
  call.eraseFromParent();

  return newInst;
}

// Default lowering of a call to the runtime function \p rtFunc.
static bool lowerCall(CallInst &call, FunctionCallee rtFunc,
                      IRBuilder<> &builder) {
  builder.SetInsertPoint(call.getIterator());
  SmallVector<Value *, 4> args = getDefaultLoweringArgs(call, rtFunc, builder);
  (void)createNewCallFor(call, rtFunc, args, builder);

  return true;
}

// Lower the thread launch intrinsic. This is a vararg intrinsic, but the
// runtime expects the variadic arguments to be bundled into a struct. We
// allocate a struct on the stack for these arguments.
//
// TODO: We should look at the number of arguments that are required and
// consider allocating a struct on the heap instead.
static bool lowerLaunchThreads(CallInst &call, IRBuilder<> &builder) {
  Function &f = *call.getFunction();
  LLVMContext &ctx = f.getContext();

  SmallVector<Value *, 4> args = getVariadicArgs(call);
  SmallVector<Type *, 4> tys;
  for (Value *arg : args)
    tys.push_back(arg->getType());
  StructType *bundleTy = StructType::get(ctx, tys, /*isPacked=*/false);

  builder.SetInsertPoint(f.getEntryBlock().begin());
  Value *bundle = builder.CreateAlloca(bundleTy);

  builder.SetInsertPoint(call.getIterator());
  for (unsigned i = 0; i < args.size(); ++i) {
    Value *off = builder.CreateConstInBoundsGEP2_32(bundleTy, bundle, 0, i);
    builder.CreateStore(args[i], off);
  }

  Module &m = *getModule(call);
  const DataLayout &dl = m.getDataLayout();
  uint32_t bundleSize = dl.getTypeStoreSize(bundleTy).getFixedValue();
  SmallVector<Value *, 4> launchArgs;
  for (unsigned i = 1; i < getNumNonVariadicArgs(call); ++i)
    launchArgs.push_back(call.getArgOperand(i));
  launchArgs.push_back(bundle);
  launchArgs.push_back(toConstant(bundleSize, ctx));

  FunctionCallee rtFunc = getRuntimeFunc(call);
  Value *newCall = createNewCallFor(call, rtFunc, launchArgs, builder);

  // The call will use the argument bundle, so it cannot be a tail call.
  cast<CallInst>(newCall)->setTailCallKind(CallInst::TCK_None);

  return true;
}

// Lower the kernel launch intrinsic. This is a vararg intrinsic, but the
// corresponding runtime functions need the arguments to be passed an array of
// pointers to the arguments. We implement this by creating a stack slot for
// each argument, and an array of pointers, each of which is a pointer to one of
// these stack slots. The runtime function is passed a pointer to this array of
// pointers.
static bool lowerLaunchKernel(CallInst &call, IRBuilder<> &builder) {
  LLVMContext &ctx = call.getContext();
  PointerType *ptrTy = PointerType::getUnqual(ctx);

  Function &f = *call.getFunction();
  BasicBlock &bbEntry = f.getEntryBlock();
  builder.SetInsertPoint(bbEntry.begin());

  SmallVector<Value *, 8> kernelArgs = getVariadicArgs(call);
  SmallVector<AllocaInst *, 8> slots;
  for (Value *kernelArg : kernelArgs)
    slots.push_back(builder.CreateAlloca(kernelArg->getType()));

  ArrayType *arrTy = ArrayType::get(ptrTy, kernelArgs.size());
  AllocaInst *argArray = builder.CreateAlloca(arrTy);

  builder.SetInsertPoint(call.getIterator());
  for (size_t i = 0; i < kernelArgs.size(); ++i) {
    builder.CreateStore(kernelArgs[i], slots[i]);
    Value *off = builder.CreateConstInBoundsGEP2_32(arrTy, argArray, 0, i);
    builder.CreateStore(slots[i], off);
  }

  SmallVector<Value *, 8> args;
  for (unsigned i = 1; i < getNumNonVariadicArgs(call); ++i)
    args.push_back(call.getArgOperand(i));
  args.push_back(argArray);

  FunctionCallee rtFunc = getRuntimeFunc(call);
  Value *newCall = createNewCallFor(call, rtFunc, args, builder);

  // The call will use the argument bundle, so it cannot be a tail call.
  cast<CallInst>(newCall)->setTailCallKind(CallInst::TCK_None);

  return true;
}

static bool lowerMobileInit(CallInst &call, IRBuilder<> &builder) {
  auto getMobileInitFunc = [](CallInst &call) -> KitFunc {
    // TODO: Currently, we always lower to a runtime function provided by
    // Kitsune that runs on the host. We should probably lower this differently
    // depending on how the buffer is being used. In some cases, it may be
    // better to do the initialization on the device.
    Value *init = call.getArgOperand(3);
    if (isBool(init))
      return KitFunc::kitrt_mobile_init_bool;
    else if (isInt8(init))
      return KitFunc::kitrt_mobile_init_i8;
    else if (isInt16(init))
      return KitFunc::kitrt_mobile_init_i16;
    else if (isInt32(init))
      return KitFunc::kitrt_mobile_init_i32;
    else if (isInt64(init))
      return KitFunc::kitrt_mobile_init_i64;
    else if (isFloat(init))
      return KitFunc::kitrt_mobile_init_float;
    else if (isDouble(init))
      return KitFunc::kitrt_mobile_init_double;
    else if (isPointer(init))
      return KitFunc::kitrt_mobile_init_from;
    else
      llvm_unreachable("Unsupported initializer type");
  };

  FunctionCallee rtFunc = getRuntimeFunc(call, getMobileInitFunc(call));
  return lowerCall(call, rtFunc, builder);
}

// Replace the Kitsune intrinsic called in the given instruction with an
// appropriate runtime function. The arguments passed to the intrinsic will
// be passed to the runtime function. Always returns true.
static bool lowerDefault(CallInst &call, IRBuilder<> &builder) {
  FunctionCallee rtFunc = getRuntimeFunc(call);
  return lowerCall(call, rtFunc, builder);
}

// The given call instruction is a call to a kitsune intrinsic. This may lower
// it (in some cases, the instruction will not be lowered - for instance if the
// the primary tapir target is one that does not permit lowering). Returns true
// if the call to the intrinsic was replaced, false otherwise.
static bool lowerKitIntrinsic(CallInst &call) {
  TTID tt = *getTTIDFromKitIntrCall(call);
  if (tt == TTID::Nolo)
    return false;

  LLVMContext &ctx = call.getContext();
  IRBuilder<> builder(ctx);

  if (requiresCustomLowering(call)) {
    switch (call.getIntrinsicID()) {
    case Intrinsic::kit_async_gpu_kernel_launch:
      return lowerLaunchKernel(call, builder);
    case Intrinsic::kit_async_cpu_threads_launch:
    case Intrinsic::kit_cpu_threads_launch:
      return lowerLaunchThreads(call, builder);
    case Intrinsic::kit_mobile_init: //
      return lowerMobileInit(call, builder);
    default:
      llvm_unreachable(
          "lowerKitIntrinsic: Intrinsic requiring custom lowering not handled");
    }
  }
  return lowerDefault(call, builder);
}

static bool lowerKitIntrinsics(Function &f) {
  // Kitsune's intrinsics cannot be invoked. The verifier will already have
  // caught this, so we only need to check for call instructions.
  SmallVector<CallInst *, 4> calls;
  for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i)
    if (auto *call = dyn_cast<CallInst>(&*i))
      if (isKitIntrinsic(call->getIntrinsicID()))
        calls.push_back(call);

  bool changed = false;
  for (CallInst *call : calls)
    changed |= lowerKitIntrinsic(*call);
  return changed;
}

namespace {

/// Pass, for the legacy pass manager, that lowers kitsune-specific intrinsics.
class LowerKitIntrinsicsLegacyPass : public FunctionPass {
public:
  LowerKitIntrinsicsLegacyPass() : FunctionPass(ID) {
    initializeLowerKitIntrinsicsLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "Lower Kitsune intrinsics"; }

  void getAnalysisUsage(AnalysisUsage &au) const override {}

  bool runOnFunction(Function &f) override { return lowerKitIntrinsics(f); }

public:
  static char ID;
};

} // namespace

char LowerKitIntrinsicsLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(LowerKitIntrinsicsLegacyPass, DEBUG_TYPE,
                      "Lower Kitsune intrinsics", false, false)
INITIALIZE_PASS_END(LowerKitIntrinsicsLegacyPass, DEBUG_TYPE,
                    "Lower Kitsune intrinsics", false, false)

FunctionPass *llvm::createLowerKitIntrinsicsLegacyPass() {
  return new LowerKitIntrinsicsLegacyPass();
}

PreservedAnalyses LowerKitIntrinsicsPass::run(Function &f,
                                              FunctionAnalysisManager &am) {
  // If any kitsune intrinsics were replaced, the call graph will have changed,
  // but other analyses will not have been invalidated.
  bool changed = lowerKitIntrinsics(f);
  if (changed) {
    PreservedAnalyses pa;
    pa.preserve<FunctionAnalysisManagerCGSCCProxy>();
    pa.preserveSet<AllAnalysesOn<Function>>();
    return pa;
  } else {
    return PreservedAnalyses::all();
  }
}
