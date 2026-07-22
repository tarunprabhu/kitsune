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
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "kit-lower-intrinsics"

using namespace llvm;

namespace {

using KitRTFuncMap = SmallDenseMap<Intrinsic::ID, KitFunc>;
using KitRTFuncArgMap = SmallDenseMap<Intrinsic::ID, SmallVector<unsigned, 4>>;

// Kitsune runtime functions for any tapir target.
static const KitRTFuncMap kitFuncs; // Currently, there are no such functions.

// Kitsune runtime functions for the cuda tapir target.
static const KitRTFuncMap kitCudaFuncs = {
    {Intrinsic::kit_async_gpu_kernel_launch, KitFunc::kitcuda_kernel_launch},
    {Intrinsic::kit_async_gpu_prefetch_dtoh, KitFunc::kitcuda_prefetch_dtoh},
    {Intrinsic::kit_async_gpu_prefetch_htod, KitFunc::kitcuda_prefetch_htod},
    {Intrinsic::kit_gpu_memcpy_dtoh, KitFunc::kitcuda_memcpy_dtoh},
    {Intrinsic::kit_gpu_memcpy_htod, KitFunc::kitcuda_memcpy_htod},
    {Intrinsic::kit_gpu_num_compute_units, KitFunc::kitcuda_num_sms},
    {Intrinsic::kit_gpu_register_devcode, KitFunc::kitcuda_register_devcode},
    {Intrinsic::kit_gpu_register_devcode_end,
     KitFunc::kitcuda_register_devcode_end},
    {Intrinsic::kit_gpu_register_global, KitFunc::kitcuda_register_global},
    {Intrinsic::kit_gpu_register_global_managed,
     KitFunc::kitcuda_register_global_managed},
    {Intrinsic::kit_gpu_stream_new, KitFunc::kitcuda_stream_new},
    {Intrinsic::kit_gpu_stream_sync, KitFunc::kitcuda_stream_sync},
    {Intrinsic::kit_gpu_symbol_address, KitFunc::kitcuda_symbol_address},
    {Intrinsic::kit_gpu_unregister_devcode,
     KitFunc::kitcuda_unregister_devcode},
    {Intrinsic::kit_mobile_alloc, KitFunc::kitcuda_managed_malloc},
    {Intrinsic::kit_mobile_free, KitFunc::kitcuda_managed_free},
    {Intrinsic::kit_runtime_finalize, KitFunc::kitcuda_finalize},
    {Intrinsic::kit_runtime_initialize, KitFunc::kitcuda_initialize},
};

// Kitsune runtime functions for the hip tapir target.
static const KitRTFuncMap kitHipFuncs = {
    {Intrinsic::kit_async_gpu_kernel_launch, KitFunc::kithip_kernel_launch},
    {Intrinsic::kit_async_gpu_prefetch_dtoh, KitFunc::kithip_prefetch_dtoh},
    {Intrinsic::kit_async_gpu_prefetch_htod, KitFunc::kithip_prefetch_htod},
    {Intrinsic::kit_gpu_memcpy_dtoh, KitFunc::kithip_memcpy_dtoh},
    {Intrinsic::kit_gpu_memcpy_htod, KitFunc::kithip_memcpy_htod},
    {Intrinsic::kit_gpu_num_compute_units, KitFunc::kithip_num_cus},
    {Intrinsic::kit_gpu_register_devcode, KitFunc::kithip_register_devcode},
    {Intrinsic::kit_gpu_register_global, KitFunc::kithip_register_global},
    {Intrinsic::kit_gpu_register_global_managed,
     KitFunc::kithip_register_global_managed},
    {Intrinsic::kit_gpu_stream_new, KitFunc::kithip_stream_new},
    {Intrinsic::kit_gpu_stream_sync, KitFunc::kithip_stream_sync},
    {Intrinsic::kit_gpu_symbol_address, KitFunc::kithip_symbol_address},
    {Intrinsic::kit_gpu_unregister_devcode, KitFunc::kithip_unregister_devcode},
    {Intrinsic::kit_mobile_alloc, KitFunc::kithip_managed_malloc},
    {Intrinsic::kit_mobile_free, KitFunc::kithip_managed_free},
    {Intrinsic::kit_runtime_finalize, KitFunc::kithip_finalize},
    {Intrinsic::kit_runtime_initialize, KitFunc::kithip_initialize},
    {Intrinsic::kit_runtime_set_xnack, KitFunc::kithip_enable_xnack},
    {Intrinsic::kit_runtime_set_y_axis_kernel_launch,
     KitFunc::kithip_enable_y_axis_launches},
};

// Kitsune runtime functions for the opencilk tapir target.
static const KitRTFuncMap kitOpenCilkFuncs = {
    {Intrinsic::kit_cpu_num_threads, KitFunc::kitocilk_num_workers},
    {Intrinsic::kit_cpu_thread_id, KitFunc::kitocilk_worker_id},
    {Intrinsic::kit_mobile_alloc, KitFunc::kitrt_malloc},
    {Intrinsic::kit_mobile_free, KitFunc::kitrt_free},
    {Intrinsic::kit_runtime_finalize, KitFunc::kitocilk_finalize},
    {Intrinsic::kit_runtime_initialize, KitFunc::kitocilk_initialize},
};

// Kitsune runtime functions for the openmp tapir target.
static const KitRTFuncMap kitOpenMPFuncs = {
    {Intrinsic::kit_cpu_num_threads, KitFunc::kitomp_num_threads},
    {Intrinsic::kit_cpu_thread_id, KitFunc::kitomp_thread_id},
    {Intrinsic::kit_cpu_threads_launch, KitFunc::kitomp_launch},
    {Intrinsic::kit_mobile_alloc, KitFunc::kitrt_malloc},
    {Intrinsic::kit_mobile_free, KitFunc::kitrt_free},
    {Intrinsic::kit_runtime_finalize, KitFunc::kitomp_finalize},
    {Intrinsic::kit_runtime_initialize, KitFunc::kitomp_initialize},
};

// Kitsune runtime functions for the pthreads tapir target.
static const KitRTFuncMap kitPthreadsFuncs = {
    {Intrinsic::kit_async_cpu_threads_launch, KitFunc::kitpthr_async_launch},
    {Intrinsic::kit_cpu_num_threads, KitFunc::kitpthr_num_threads},
    {Intrinsic::kit_cpu_thread_id, KitFunc::kitpthr_thread_id},
    {Intrinsic::kit_cpu_threads_sync, KitFunc::kitpthr_sync},
    {Intrinsic::kit_mobile_alloc, KitFunc::kitrt_malloc},
    {Intrinsic::kit_mobile_free, KitFunc::kitrt_free},
    {Intrinsic::kit_runtime_finalize, KitFunc::kitpthr_finalize},
    {Intrinsic::kit_runtime_initialize, KitFunc::kitpthr_initialize},
};

// Kitsune runtime functions for the qthreads tapir target.
static const KitRTFuncMap kitQthreadsFuncs = {
    {Intrinsic::kit_cpu_num_threads, KitFunc::kitqthr_num_workers},
    {Intrinsic::kit_cpu_thread_id, KitFunc::kitqthr_worker_id},
    {Intrinsic::kit_cpu_threads_launch, KitFunc::kitqthr_launch},
    // There may be some benefit to using the memory allocation functions
    // provided by qthreads. Those use memory pools and it is not yet clear if
    // that is something we should consider using.
    {Intrinsic::kit_mobile_alloc, KitFunc::kitrt_malloc},
    {Intrinsic::kit_mobile_free, KitFunc::kitrt_free},
    {Intrinsic::kit_runtime_finalize, KitFunc::kitqthr_finalize},
    {Intrinsic::kit_runtime_initialize, KitFunc::kitqthr_initialize},
};

// Kitsune runtime functions for the serial tapir target.
static const KitRTFuncMap kitSerialFuncs = {
    {Intrinsic::kit_cpu_thread_id, KitFunc::kitser_thread_id},
    {Intrinsic::kit_mobile_alloc, KitFunc::kitrt_malloc},
    {Intrinsic::kit_mobile_free, KitFunc::kitrt_free},
    {Intrinsic::kit_runtime_finalize, KitFunc::kitser_finalize},
    {Intrinsic::kit_runtime_initialize, KitFunc::kitser_initialize},
};

// Runtime library function maps for tapir targets that have a corresponding
// kitsune runtime.
static const SmallDenseMap<TTID, KitRTFuncMap> kitTTFuncs = {
    {TTID::Cuda, kitCudaFuncs},         {TTID::Hip, kitHipFuncs},
    {TTID::OpenCilk, kitOpenCilkFuncs}, {TTID::OpenMP, kitOpenMPFuncs},
    {TTID::Pthreads, kitPthreadsFuncs}, {TTID::Qthreads, kitQthreadsFuncs},
    {TTID::Serial, kitSerialFuncs},
};

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

static FunctionCallee getOrInsertLibFunc(Module &m, TTID tt, Intrinsic::ID id) {
  assert(kitTTFuncs.find(tt) != kitTTFuncs.end() &&
         "getRuntimeFunc: Invalid tapir target for intrinsic");
  const KitRTFuncMap &funcs = kitTTFuncs.at(tt);

  assert(funcs.find(id) != funcs.end() &&
         "getRuntimeFunc: No kitsune library function for tapir target");
  return getOrInsertLibFunc(m, funcs.at(id));
}

// Get the kitsune runtime function that will replace the intrinsic called in
// the given call instruction.
static FunctionCallee getRuntimeFunc(CallInst &call) {
  auto getMobileInitFunc = [](CallInst &call) -> KitFunc {
    // Currently, we always lower to a runtime function provided by Kitsune that
    // runs on the host.
    // FIXME: We should probably lower this differently depending on how the
    // buffer is being used. In some cases, it may be better to do the
    // initialization on the device.
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

  Intrinsic::ID id = call.getIntrinsicID();
  Module &m = *call.getModule();
  switch (id) {
  case Intrinsic::kit_mobile_init:
    return getOrInsertLibFunc(m, getMobileInitFunc(call));
  default:
    return getOrInsertLibFunc(m, *getTTIDFromKitIntrCall(call), id);
  }
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
                               ArrayRef<Value *> origArgs) {
  auto requiresMobilePointerCast = [](Type *src, Type *dst) -> bool {
    return (isMobilePointerTy(src) && !isMobilePointerTy(dst)) ||
           (isMobilePointerTy(dst) && !isMobilePointerTy(src));
  };

  auto requiresCastFromBool = [](Type *src, Type *dst) -> bool {
    return src->isIntegerTy(1) && dst->isIntegerTy();
  };

  // In most cases, we expect the intrinsics and their corresponding runtime
  // functions to have exactly the same signature. There are a limited number of
  // cases where we permit casting though.
  auto maybeCast = [&](Value *v, Type *dstTy, IRBuilder<> &builder) -> Value * {
    Type *srcTy = v->getType();
    if (requiresMobilePointerCast(srcTy, dstTy))
      return builder.CreateAddrSpaceCast(v, dstTy);
    else if (requiresCastFromBool(srcTy, dstTy))
      return builder.CreateZExt(v, dstTy, /*name=*/"", /*isNonNeg=*/true);
    return v;
  };

  LLVMContext &ctx = call.getContext();
  IRBuilder<> builder(ctx);

  builder.SetInsertPoint(call.getIterator());

  SmallVector<Value *, 8> args;
  FunctionType *funcType = f.getFunctionType();
  for (unsigned i = 0; i < origArgs.size(); ++i)
    args.push_back(maybeCast(origArgs[i], funcType->getParamType(i), builder));

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

  Value *newInst = maybeCast(newCall, call.getType(), builder);

  call.replaceAllUsesWith(newInst);
  call.eraseFromParent();

  return newInst;
}

// Lower the thread launch intrinsic. This is a vararg intrinsic, but the
// runtime expects the variadic arguments to be bundled into a struct. We
// allocate a struct on the stack for these arguments.
//
// TODO: We should look at the number of arguments that are required and
// consider allocating a struct on the heap instead.
static bool lowerLaunchThreads(CallInst &call) {
  Function &f = *call.getFunction();
  LLVMContext &ctx = f.getContext();
  IRBuilder<> builder(ctx);

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
  uint64_t bundleSize = dl.getTypeStoreSize(bundleTy).getFixedValue();
  SmallVector<Value *, 4> launchArgs;
  for (unsigned i = 1; i < getNumNonVariadicArgs(call); ++i)
    launchArgs.push_back(call.getArgOperand(i));
  launchArgs.push_back(bundle);
  launchArgs.push_back(toConstant(bundleSize, ctx));

  FunctionCallee rtFunc = getRuntimeFunc(call);
  assert(rtFunc && "Got runtime function for intrinsic call");

  // The call will use the argument bundle, so it cannot be a tail call.
  Value *newCall = createNewCallFor(call, rtFunc, launchArgs);
  cast<CallInst>(newCall)->setTailCallKind(CallInst::TCK_None);

  return true;
}

// Lower the kernel launch intrinsic. This is a vararg intrinsic, but the
// corresponding runtime functions need the arguments to be passed an array of
// pointers to the arguments. We implement this by creating a stack slot for
// each argument, and an array of pointers, each of which is a pointer to one of
// these stack slots. The runtime function is passed a pointer to this array of
// pointers.
static bool lowerLaunchKernel(CallInst &call) {
  LLVMContext &ctx = call.getContext();
  PointerType *ptrTy = PointerType::getUnqual(ctx);

  IRBuilder<> builder(ctx);
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
  assert(rtFunc && "Got runtime function for intrinsic call");

  // The call will use the argument bundle, so it cannot be a tail call.
  Value *newCall = createNewCallFor(call, rtFunc, args);
  cast<CallInst>(newCall)->setTailCallKind(CallInst::TCK_None);

  return true;
}

// Replace the Kitsune intrinsic called in the given instruction with an
// appropriate runtime function. The arguments passed to the intrinsic will
// be passed to the runtime function. Always returns true.
static bool lowerDefault(CallInst &call) {
  // The first argument will be the TTID. Everything else will be passed along
  // to the lowered function.
  SmallVector<Value *, 4> args;
  for (unsigned i = 1; i < call.arg_size(); ++i)
    args.push_back(call.getArgOperand(i));

  FunctionCallee rtFunc = getRuntimeFunc(call);
  assert(rtFunc && "Got runtime function for intrinsic call");

  (void)createNewCallFor(call, rtFunc, args);

  return true;
}

// The given call instruction is a call to a kitsune intrinsic. This may lower
// it (in some cases, the instruction will not be lowered - for instance if the
// the primary tapir target is one that does not permit lowering). Returns true
// if the call to the intrinsic was replaced, false otherwise.
static bool lowerKitIntrinsic(CallInst &call) {
  TTID tt = *getTTIDFromKitIntrCall(call);
  if (tt == TTID::Nolo)
    return false;

  switch (call.getIntrinsicID()) {
  case Intrinsic::kit_async_gpu_kernel_launch:
    return lowerLaunchKernel(call);
  case Intrinsic::kit_async_cpu_threads_launch:
  case Intrinsic::kit_cpu_threads_launch:
    return lowerLaunchThreads(call);
  default:
    return lowerDefault(call);
  }
}

static bool lowerKitIntrinsics(Function &f) {
  SmallVector<CallInst *, 4> calls;
  for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i) {
    if (auto *call = dyn_cast<CallInst>(&*i)) {
      if (isKitIntrinsic(call->getIntrinsicID()))
        calls.push_back(call);
    } else if (auto *invoke = dyn_cast<InvokeInst>(&*i)) {
      // TODO: This invoke check should be moved to the verifier.
      if (isKitIntrinsic(invoke->getIntrinsicID()))
        llvm_unreachable("Invoke of kitsune intrinsic");
    }
  }

  bool changed = false;
  for (CallInst *call : calls)
    changed |= lowerKitIntrinsic(*call);
  return changed;
}

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
