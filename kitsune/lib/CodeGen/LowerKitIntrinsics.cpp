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
// Most of these are lowered to calls to functions from Kitsune's runtime. But
// this is not always the case.
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

/// Kitsune runtime functions for any tapir target.
static const KitRTFuncMap kitFuncs; // Currently, there are no such functions.

/// Kitsune runtime functions for the cuda tapir target.
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

/// Kitsune runtime functions for the hip tapir target.
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

/// Kitsune runtime functions for the opencilk tapir target.
static const KitRTFuncMap kitOpenCilkFuncs = {
    {Intrinsic::kit_cpu_num_threads, KitFunc::kitocilk_num_workers},
    {Intrinsic::kit_cpu_thread_id, KitFunc::kitocilk_worker_id},
    {Intrinsic::kit_mobile_alloc, KitFunc::kitrt_malloc},
    {Intrinsic::kit_mobile_free, KitFunc::kitrt_free},
    {Intrinsic::kit_runtime_finalize, KitFunc::kitocilk_finalize},
    {Intrinsic::kit_runtime_initialize, KitFunc::kitocilk_initialize},
};

/// Kitsune runtime functions for the openmp tapir target.
static const KitRTFuncMap kitOpenMPFuncs = {
    {Intrinsic::kit_cpu_num_threads, KitFunc::kitomp_num_threads},
    {Intrinsic::kit_cpu_thread_id, KitFunc::kitomp_thread_id},
    {Intrinsic::kit_cpu_threads_launch, KitFunc::kitomp_launch},
    {Intrinsic::kit_mobile_alloc, KitFunc::kitrt_malloc},
    {Intrinsic::kit_mobile_free, KitFunc::kitrt_free},
    {Intrinsic::kit_runtime_finalize, KitFunc::kitomp_finalize},
    {Intrinsic::kit_runtime_initialize, KitFunc::kitomp_initialize},
};

/// Kitsune runtime functions for the pthreads tapir target.
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

/// Kitsune runtime functions for the qthreads tapir target.
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

/// Kitsune runtime functions for the serial tapir target.
static const KitRTFuncMap kitSerialFuncs = {
    {Intrinsic::kit_cpu_thread_id, KitFunc::kitser_thread_id},
    {Intrinsic::kit_mobile_alloc, KitFunc::kitrt_malloc},
    {Intrinsic::kit_mobile_free, KitFunc::kitrt_free},
    {Intrinsic::kit_runtime_finalize, KitFunc::kitser_finalize},
    {Intrinsic::kit_runtime_initialize, KitFunc::kitser_initialize},
};

/// Runtime library function maps for tapir targets that have a corresponding
/// kitsune runtime.
static const SmallDenseMap<TTID, KitRTFuncMap> kitTTFuncs = {
    {TTID::Cuda, kitCudaFuncs},         {TTID::Hip, kitHipFuncs},
    {TTID::OpenCilk, kitOpenCilkFuncs}, {TTID::OpenMP, kitOpenMPFuncs},
    {TTID::Pthreads, kitPthreadsFuncs}, {TTID::Qthreads, kitQthreadsFuncs},
    {TTID::Serial, kitSerialFuncs},
};

/// Return a new attribute list which is exactly the same as the given
/// attribute list \ref attrs except that the attributes at index \ref src of
/// \ref call's attribute list are added to index \ref dst of \ref attrs. The
/// newly created attribute list is returned.
static AttributeList addAttrsFrom(AttributeList attrs, unsigned dst,
                                  const CallInst &call, unsigned src) {
  LLVMContext &ctx = call.getContext();
  AttributeList callAttrs = call.getAttributes();
  for (const Attribute &attr : callAttrs.getAttributes(src))
    attrs = attrs.addAttributeAtIndex(ctx, dst, attr);
  return attrs;
}

/// Return a new attribute list which is exactly the same as the given attribute
/// list \ref attrs except that the attributes at index \ref src of \ref call's
/// attribute list are added to index \ref src of \ref attrs. The newly created
/// attribute list is returned.
static AttributeList addAttrsFrom(AttributeList attrs, const CallInst &call,
                                  unsigned src) {
  return addAttrsFrom(attrs, src, call, src);
}

/// Create a new attribute list that will eventually be applied to the
/// replacement of \p call. \p call is expected to be a direct call to a
/// Kitsune-specific intrinsic. Since the first argument to such intrinsics will
/// always be a TTID, that is skipped. The remaining non-variadic arguments are
/// assumed to be passed as-is to the new call, so their attributes are copied.
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

/// All runtime intrinsics take the TTID as the first argument. Parse this into
/// a TTID enum.
static TTID getTTID(Value *v) {
  assert(isa<Constant>(v) && "TTID must be a constant");
  assert(fromConstant<TTID>(*cast<Constant>(v)) && "Not a valid TTID");

  return *fromConstant<TTID>(*cast<Constant>(v));
}

/// Main implementation class to lower Kitsune intrinsics.
class LowerKitIntrinsics {
private:
  FunctionCallee getOrInsertLibFunc(Module &m, Intrinsic::ID id) {
    assert(kitFuncs.find(id) != kitFuncs.end() &&
           "getRuntimeFunc: No kitsune library function for intrinsic");
    return ::getOrInsertLibFunc(m, kitFuncs.at(id));
  }

  FunctionCallee getOrInsertLibFunc(Module &m, TTID tt, Intrinsic::ID id) {
    assert(kitTTFuncs.find(tt) != kitTTFuncs.end() &&
           "getRuntimeFunc: Invalid tapir target for intrinsic");
    const KitRTFuncMap &funcs = kitTTFuncs.at(tt);

    assert(funcs.find(id) != funcs.end() &&
           "getRuntimeFunc: No kitsune library function for tapir target");
    return ::getOrInsertLibFunc(m, funcs.at(id));
  }

  FunctionCallee getMobileAllocFunc(Module &m, CallInst &call) {
    Intrinsic::ID id = call.getIntrinsicID();
    TTID tt = *getTTIDFromKitIntrCall(call);

    switch (tt) {
    case TTID::Nolo:
      return nullptr;
    case TTID::Cuda:
    case TTID::Hip:
    case TTID::OpenCilk:
    case TTID::OpenMP:
    case TTID::Pthreads:
    case TTID::Qthreads:
    case TTID::Serial:
      return getOrInsertLibFunc(m, tt, id);
    case TTID::Custom:
      // TODO: A custom tapir target may require a custom memory allocator.
      // Currently, there is no way to have the plugin specify a memory
      // allocator to use, so just default to using libc's malloc.
      return ::getOrInsertLibFunc(m, KitFunc::kitrt_malloc);
    case TTID::Lambda:
    case TTID::OMPTask:
    case TTID::Realm:
      // These tapir targets are not fully supported yet, but add them to this
      // switch to ensure that a warning is emitted when a new tapir target is
      // added.
      break;
    }
    llvm_unreachable("getMobileAllocFunc: TTID not handled");
  }

  FunctionCallee getMobileFreeFunc(Module &m, CallInst &call) {
    Intrinsic::ID id = call.getIntrinsicID();
    TTID tt = *getTTIDFromKitIntrCall(call);

    switch (tt) {
    case TTID::Nolo:
      return nullptr;
    case TTID::Cuda:
    case TTID::Hip:
    case TTID::OpenCilk:
    case TTID::OpenMP:
    case TTID::Pthreads:
    case TTID::Qthreads:
    case TTID::Serial:
      return getOrInsertLibFunc(m, tt, id);
    case TTID::Custom:
      // TODO: A custom tapir target may require a custom memory deallocator.
      // Currently, there is no way to have the plugin specify a memory
      // deallocator to use, so just default to using libc's free.
      return ::getOrInsertLibFunc(m, KitFunc::kitrt_free);
    case TTID::Lambda:
    case TTID::OMPTask:
    case TTID::Realm:
      // These tapir targets are not fully supported yet, but add them to this
      // switch to ensure that a warning is emitted when a new tapir target is
      // added.
      break;
    }
    llvm_unreachable("getMobileFreeFunc: TTID not handled");
  }

  FunctionCallee getMobileInitFunc(Module &m, CallInst &call) {
    TTID tt = *getTTIDFromKitIntrCall(call);
    if (tt == TTID::Nolo)
      return nullptr;

    // Currently, we always lower to a runtime function provided by Kitsune that
    // runs on the host.
    // FIXME: We should probably lower this differently depending on how the
    // buffer is being used. In some cases, it may be better to do the
    // initialization on the device.
    Value *init = call.getArgOperand(3);
    if (isBool(init))
      return ::getOrInsertLibFunc(m, KitFunc::kitrt_mobile_init_bool);
    else if (isInt8(init))
      return ::getOrInsertLibFunc(m, KitFunc::kitrt_mobile_init_i8);
    else if (isInt16(init))
      return ::getOrInsertLibFunc(m, KitFunc::kitrt_mobile_init_i16);
    else if (isInt32(init))
      return ::getOrInsertLibFunc(m, KitFunc::kitrt_mobile_init_i32);
    else if (isInt64(init))
      return ::getOrInsertLibFunc(m, KitFunc::kitrt_mobile_init_i64);
    else if (isFloat(init))
      return ::getOrInsertLibFunc(m, KitFunc::kitrt_mobile_init_float);
    else if (isDouble(init))
      return ::getOrInsertLibFunc(m, KitFunc::kitrt_mobile_init_double);
    else if (isPointer(init))
      return ::getOrInsertLibFunc(m, KitFunc::kitrt_mobile_init_from);
    else
      llvm_unreachable("Unsupported initializer type");
  }

  /// Get the kitsune runtime function that will replace the intrinsic called in
  /// the given call instruction.
  FunctionCallee getRuntimeFunc(CallInst &call) {
    Intrinsic::ID id = call.getIntrinsicID();
    Module &m = *call.getModule();

    switch (id) {
    case Intrinsic::kit_mobile_alloc:
      return getMobileAllocFunc(m, call);
    case Intrinsic::kit_mobile_free:
      return getMobileFreeFunc(m, call);
    case Intrinsic::kit_mobile_init:
      return getMobileInitFunc(m, call);
    default:
      return getOrInsertLibFunc(m, getTTID(call.getArgOperand(0)), id);
    }
  }

  /// Create a new call to the given function to replace an existing call. The
  /// debug info, metadata, calling convention and tail call kind will be copied
  /// over from the original call. However, the attributes will not be copied.
  /// The new call is returned, but the original call will remain unchanged.
  ///
  /// The attributes are not copied because there are some intrinsics where the
  /// attributes cannot be copied over directly. To avoid having conditional
  /// statements in this function, we require the attributes to be copied over
  /// by callers.
  CallInst *createNewCallFor(CallInst &call, FunctionCallee f,
                             ArrayRef<Value *> args) {
    StringRef name = call.getName();
    BasicBlock::iterator pos = call.getIterator();

    CallInst *newCall = CallInst::Create(f, args, name, pos);
    newCall->cloneDebugInfoFrom(&call);
    newCall->copyMetadata(call);
    newCall->setCallingConv(call.getCallingConv());
    newCall->takeName(&call);

    // Because the result of the lowered intrinsic may be cast to a different
    // type (typically this will be an address space cast), tail calls cannot be
    // guaranteed.
    CallInst::TailCallKind tck = call.getTailCallKind();
    if (tck == CallInst::TCK_MustTail)
      newCall->setTailCallKind(CallInst::TCK_Tail);
    else
      newCall->setTailCallKind(tck);

    return newCall;
  }

  /// Depending on the tapir target to be used, this may not replace the call.
  /// In that case, return false. Otherwise, return true.
  bool lowerMobileAlloc(CallInst &call) {
    FunctionCallee f = getRuntimeFunc(call);
    if (not f.getCallee())
      return false;

    BasicBlock::iterator pos = call.getIterator();
    Type *retTy = call.getType();

    // The result of the new call will be a pointer in the default address
    // space. However, all uses of the call will be in the mobile address space.
    CallInst *newCall = createNewCallFor(call, f, call.getArgOperand(1));
    CastInst *cst =
        CastInst::Create(Instruction::AddrSpaceCast, newCall, retTy, "", pos);
    newCall->setAttributes(createNewAttrList(call));

    cst->moveAfter(newCall);
    call.replaceAllUsesWith(cst);
    call.eraseFromParent();

    return true;
  }

  /// Depending on the tapir target to be used, this may not replace the call.
  /// In the case, return false, otherwise, return true.
  bool lowerMobileFree(CallInst &call) {
    FunctionCallee f = getRuntimeFunc(call);
    if (not f.getCallee())
      return false;

    LLVMContext &ctx = call.getContext();
    Type *ptrTy = PointerType::getUnqual(ctx);
    BasicBlock::iterator pos = call.getIterator();

    // The call expects a pointer in the default address space, but the
    // argument to Kitsune's intrinsic will be in the mobile address space.
    CastInst *cst = CastInst::Create(Instruction::AddrSpaceCast,
                                     call.getArgOperand(1), ptrTy, "", pos);
    CallInst *newCall = createNewCallFor(call, f, cst);
    newCall->setAttributes(createNewAttrList(call));

    call.replaceAllUsesWith(newCall);
    call.eraseFromParent();

    return true;
  }

  bool lowerMobileInit(CallInst &call) {
    auto getBuffer = [](CallInst &call) -> Value * {
      // The call expects the pointer to the buffer to be in the default address
      // space, but the argument to Kitsune's intrinsic will be in the mobile
      // address space.
      LLVMContext &ctx = call.getContext();
      Type *ptr = PointerType::getUnqual(ctx);
      BasicBlock::iterator pos = call.getIterator();
      Value *buf = call.getArgOperand(1);
      return CastInst::Create(Instruction::AddrSpaceCast, buf, ptr, "", pos);
    };

    auto getInit = [](CallInst &call) -> Value * {
      Value *init = call.getArgOperand(3);
      if (!isBool(init))
        return init;

      LLVMContext &ctx = call.getContext();
      Type *i8 = Type::getInt8Ty(ctx);
      BasicBlock::iterator pos = call.getIterator();
      return CastInst::Create(Instruction::ZExt, init, i8, "", pos);
    };

    FunctionCallee f = getRuntimeFunc(call);
    if (not f.getCallee())
      return false;

    Value *buf = getBuffer(call);
    Value *n = call.getArgOperand(2);
    Value *init = getInit(call);
    SmallVector<Value *, 4> args = {buf, n, init};
    if (call.arg_size() > 4)
      args.push_back(call.getArgOperand(4));

    CallInst *newCall = createNewCallFor(call, f, args);
    newCall->setAttributes(createNewAttrList(call));

    call.replaceAllUsesWith(newCall);
    call.eraseFromParent();

    return true;
  }

  /// Lower the thread launch intrinsic. This is a vararg intrinsic, but the
  /// runtime expects the variadic arguments to be bundled into a struct. We
  /// allocate a struct on the stack for these arguments.
  ///
  /// TODO: We should look at the number of arguments that are required and
  /// consider allocating a struct on the heap instead.
  bool lowerLaunchThreads(CallInst &call) {
    Module &m = *getModule(call);
    LLVMContext &ctx = m.getContext();
    Type *i32 = Type::getInt32Ty(ctx);
    Type *i64 = Type::getInt64Ty(ctx);

    SmallVector<Value *, 4> args = getVariadicArgs(call);

    SmallVector<Type *, 4> tys;
    for (Value *arg : args)
      tys.push_back(arg->getType());
    StructType *bundleTy = StructType::get(ctx, tys, /*isPacked=*/false);

    BasicBlock &bbEntry = call.getFunction()->getEntryBlock();
    Value *bundle =
        new AllocaInst(bundleTy, /*addrspace=*/0, "", bbEntry.begin());
    Constant *zero = ConstantInt::get(i32, 0, /*isSigned=*/false);
    for (size_t i = 0; i < args.size(); ++i) {
      Constant *idx = ConstantInt::get(i32, i, /*isSigned=*/false);
      GetElementPtrInst *off = GetElementPtrInst::CreateInBounds(
          bundleTy, bundle, {zero, idx}, "", call.getIterator());
      (void)new StoreInst(args[i], off, call.getIterator());
    }

    const DataLayout &dl = m.getDataLayout();
    uint64_t bundleSize = dl.getTypeStoreSize(bundleTy).getFixedValue();
    SmallVector<Value *, 4> launchArgs;
    for (unsigned i = 1; i < getNumNonVariadicArgs(call); ++i)
      launchArgs.push_back(call.getArgOperand(i));
    launchArgs.push_back(bundle);
    launchArgs.push_back(ConstantInt::get(i64, bundleSize));

    FunctionCallee rtFunc = getRuntimeFunc(call);
    CallInst *newCall = createNewCallFor(call, rtFunc, launchArgs);

    // The call will use the argument bundle, so it cannot be a tail call.
    newCall->setAttributes(createNewAttrList(call));
    newCall->setTailCallKind(CallInst::TCK_None);

    call.replaceAllUsesWith(newCall);
    call.eraseFromParent();

    return true;
  }

  /// Lower the kernel launch intrinsic. This is a vararg intrinsic, but the
  /// corresponding runtime functions need the arguments to be passed an array
  /// of pointers to the arguments. We implement this by creating a stack slot
  /// for each argument, and an array of pointers, each of which is a pointer to
  /// one of these stack slots. The runtime function is passed a pointer to
  /// this array of pointers.
  bool lowerLaunchKernel(CallInst &call) {
    LLVMContext &ctx = call.getContext();
    Type *i64 = Type::getInt64Ty(ctx);
    PointerType *ptrTy = PointerType::getUnqual(ctx);
    Constant *c0 = ConstantInt::get(i64, 0);

    BasicBlock &bbEntry = call.getFunction()->getEntryBlock();
    SmallVector<Value *, 8> kernelArgs = getVariadicArgs(call);
    ArrayType *arrTy = ArrayType::get(ptrTy, kernelArgs.size());
    AllocaInst *argArray =
        new AllocaInst(arrTy, /*addrspace=*/0, "", bbEntry.begin());
    for (size_t i = 0; i < kernelArgs.size(); ++i) {
      Constant *ci = ConstantInt::get(i64, i);
      Value *indices[] = {c0, ci};
      Value *kernelArg = kernelArgs[i];
      Type *argTy = kernelArg->getType();

      AllocaInst *slot =
          new AllocaInst(argTy, /*addrspace=*/0, "", argArray->getIterator());
      (void)new StoreInst(kernelArg, slot, call.getIterator());
      GetElementPtrInst *argOffset = GetElementPtrInst::CreateInBounds(
          arrTy, argArray, indices, "", call.getIterator());
      (void)new StoreInst(slot, argOffset, call.getIterator());
    }

    SmallVector<Value *, 8> args;
    for (unsigned i = 1; i < getNumNonVariadicArgs(call); ++i)
      args.push_back(call.getArgOperand(i));
    args.push_back(argArray);

    FunctionCallee rtFunc = getRuntimeFunc(call);
    CallInst *newCall = createNewCallFor(call, rtFunc, args);

    // The call will use the argument bundle, so it cannot be a tail call.
    newCall->setAttributes(createNewAttrList(call));
    newCall->setTailCallKind(CallInst::TCK_None);

    call.replaceAllUsesWith(newCall);
    call.eraseFromParent();

    return true;
  }

  /// Replace the Kitsune intrinsic called in the given instruction with an
  /// appropriate runtime function. The arguments passed to the intrinsic will
  /// be passed to the runtime function. Always returns true.
  bool lowerIntrinsicDefault(CallInst &call) {
    // The first argument will be the TTID. Everything else will be passed along
    // to the lowered function.
    SmallVector<Value *, 4> args;
    for (unsigned i = 1; i < call.arg_size(); ++i)
      args.push_back(call.getArgOperand(i));

    FunctionCallee f = getRuntimeFunc(call);
    CallInst *newCall = createNewCallFor(call, f, args);
    newCall->setAttributes(createNewAttrList(call));

    call.replaceAllUsesWith(newCall);
    call.eraseFromParent();

    return true;
  }

  /// The given call instruction is a call to a kitsune intrinsic. This may
  /// lower it (in some cases, the instruction will not be lowered - for
  /// instance if the primary tapir target is one that does not permit
  /// lowering). Returns true if the call to the intrinsic was replaced, false
  /// otherwise.
  bool lowerIntrinsic(CallInst &call) {
    switch (call.getIntrinsicID()) {
    case Intrinsic::kit_mobile_alloc:
      return lowerMobileAlloc(call);
    case Intrinsic::kit_mobile_free:
      return lowerMobileFree(call);
    case Intrinsic::kit_mobile_init:
      return lowerMobileInit(call);
    case Intrinsic::kit_async_gpu_kernel_launch:
      return lowerLaunchKernel(call);
    case Intrinsic::kit_async_cpu_threads_launch:
    case Intrinsic::kit_cpu_threads_launch:
      return lowerLaunchThreads(call);
    default:
      return lowerIntrinsicDefault(call);
    }
  }

public:
  bool run(Function &f) {
    SmallVector<CallInst *, 4> calls;
    for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i) {
      if (auto *call = dyn_cast<CallInst>(&*i)) {
        if (isKitIntrinsic(call->getIntrinsicID()))
          calls.push_back(call);
      } else if (auto *invoke = dyn_cast<InvokeInst>(&*i)) {
        if (isKitIntrinsic(invoke->getIntrinsicID()))
          llvm_unreachable("Invoke of kitsune intrinsic");
      }
    }

    bool changed = false;
    for (CallInst *call : calls)
      changed |= lowerIntrinsic(*call);
    return changed;
  }
};

/// Pass, for the legacy pass manager, that lowers kitsune-specific intrinsics.
class LowerKitIntrinsicsLegacyPass : public FunctionPass {
public:
  LowerKitIntrinsicsLegacyPass() : FunctionPass(ID) {
    initializeLowerKitIntrinsicsLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "Lower Kitsune intrinsics"; }

  void getAnalysisUsage(AnalysisUsage &au) const override {}

  bool runOnFunction(Function &f) override {
    return LowerKitIntrinsics().run(f);
  }

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
  bool changed = LowerKitIntrinsics().run(f);
  if (changed) {
    PreservedAnalyses pa;
    pa.preserve<FunctionAnalysisManagerCGSCCProxy>();
    pa.preserveSet<AllAnalysesOn<Function>>();
    return pa;
  } else {
    return PreservedAnalyses::all();
  }
}
