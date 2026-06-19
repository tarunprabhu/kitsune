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
#include "kitsune/Core/IntrinsicUtils.h"
#include "kitsune/Core/Tapir.h"
#include "kitsune/Core/ValueUtils.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"

#define DEBUG_TYPE "kit-lower-intrinsics"

using namespace llvm;

namespace {

using KitRTFuncMap = SmallDenseMap<Intrinsic::ID, LibFunc>;
using KitRTFuncArgMap = SmallDenseMap<Intrinsic::ID, SmallVector<unsigned, 4>>;

/// Kitsune runtime functions for any tapir target.
static const KitRTFuncMap kitFuncs = {
    {Intrinsic::kit_runtime_set_verbose, LibFunc_kitrt_enable_verbose},
};

/// Kitsune runtime functions for the cuda tapir target.
static const KitRTFuncMap kitCudaFuncs = {
    {Intrinsic::kit_async_gpu_kernel_launch, LibFunc_kitcuda_launch_kernel},
    {Intrinsic::kit_async_gpu_prefetch_dtoh, LibFunc_kitcuda_prefetch_dtoh},
    {Intrinsic::kit_async_gpu_prefetch_htod, LibFunc_kitcuda_prefetch_htod},
    {Intrinsic::kit_gpu_stream_new, LibFunc_kitcuda_get_thread_stream},
    {Intrinsic::kit_gpu_stream_sync, LibFunc_kitcuda_sync_stream},
    {Intrinsic::kit_gpu_symbol_address, LibFunc_kitcuda_symbol_device_ptr},
    {Intrinsic::kit_gpu_symbol_memcpy_dtoh, LibFunc_kitcuda_symbol_memcpy_dtoh},
    {Intrinsic::kit_gpu_symbol_memcpy_htod, LibFunc_kitcuda_symbol_memcpy_htod},
    {Intrinsic::kit_mobile_alloc, LibFunc_kitcuda_managed_malloc},
    {Intrinsic::kit_mobile_free, LibFunc_kitcuda_managed_free},
    {Intrinsic::kit_reduce_num_partials, LibFunc_kitcuda_reduce_num_partials},
    {Intrinsic::kit_runtime_finalize, LibFunc_kitcuda_finalize},
    {Intrinsic::kit_runtime_initialize, LibFunc_kitcuda_initialize},
    {Intrinsic::kit_runtime_set_fixed_tpb, LibFunc_kitcuda_set_fixed_tpb},
    {Intrinsic::kit_runtime_set_max_tpb, LibFunc_kitcuda_set_max_tpb},
    {Intrinsic::kit_runtime_set_kernel_launch_refinement,
     LibFunc_kitcuda_enable_refine_launches},
};

/// Kitsune runtime functions for the hip tapir target.
static const KitRTFuncMap kitHipFuncs = {
    {Intrinsic::kit_async_gpu_kernel_launch, LibFunc_kithip_launch_kernel},
    {Intrinsic::kit_async_gpu_prefetch_dtoh, LibFunc_kithip_prefetch_dtoh},
    {Intrinsic::kit_async_gpu_prefetch_htod, LibFunc_kithip_prefetch_htod},
    {Intrinsic::kit_gpu_stream_new, LibFunc_kithip_get_thread_stream},
    {Intrinsic::kit_gpu_stream_sync, LibFunc_kithip_sync_stream},
    {Intrinsic::kit_gpu_symbol_address, LibFunc_kithip_symbol_device_ptr},
    {Intrinsic::kit_gpu_symbol_memcpy_dtoh, LibFunc_kithip_symbol_memcpy_dtoh},
    {Intrinsic::kit_gpu_symbol_memcpy_htod, LibFunc_kithip_symbol_memcpy_htod},
    {Intrinsic::kit_mobile_alloc, LibFunc_kithip_managed_malloc},
    {Intrinsic::kit_mobile_free, LibFunc_kithip_managed_free},
    {Intrinsic::kit_reduce_num_partials, LibFunc_kithip_reduce_num_partials},
    {Intrinsic::kit_runtime_finalize, LibFunc_kithip_finalize},
    {Intrinsic::kit_runtime_initialize, LibFunc_kithip_initialize},
    {Intrinsic::kit_runtime_set_fixed_tpb, LibFunc_kithip_set_fixed_tpb},
    {Intrinsic::kit_runtime_set_max_tpb, LibFunc_kithip_set_max_tpb},
    {Intrinsic::kit_runtime_set_xnack, LibFunc_kithip_enable_xnack},
    {Intrinsic::kit_runtime_set_y_axis_kernel_launch,
     LibFunc_kithip_enable_y_axis_launches},
};

/// Kitsune runtime functions for the opencilk tapir target.
static const KitRTFuncMap kitOpenCilkFuncs = {
    {Intrinsic::kit_mobile_alloc, LibFunc_malloc},
    {Intrinsic::kit_mobile_free, LibFunc_free},
    {Intrinsic::kit_reduce_num_partials, LibFunc_kitocilk_reduce_num_partials},
    {Intrinsic::kit_runtime_finalize, LibFunc_kitocilk_finalize},
    {Intrinsic::kit_runtime_initialize, LibFunc_kitocilk_initialize},
};

/// Kitsune runtime functions for the openmp tapir target.
static const KitRTFuncMap kitOpenMPFuncs = {
    {Intrinsic::kit_mobile_alloc, LibFunc_malloc},
    {Intrinsic::kit_mobile_free, LibFunc_free},
    {Intrinsic::kit_cpu_threads_launch, LibFunc_kitomp_launch},
    {Intrinsic::kit_reduce_num_partials, LibFunc_kitomp_reduce_num_partials},
    {Intrinsic::kit_runtime_finalize, LibFunc_kitomp_finalize},
    {Intrinsic::kit_runtime_initialize, LibFunc_kitomp_initialize},
};

/// Kitsune runtime functions for the pthreads tapir target.
static const KitRTFuncMap kitPthreadsFuncs = {
    {Intrinsic::kit_async_cpu_threads_launch, LibFunc_kitpthr_launch},
    {Intrinsic::kit_cpu_threads_sync, LibFunc_kitpthr_sync},
    {Intrinsic::kit_mobile_alloc, LibFunc_malloc},
    {Intrinsic::kit_mobile_free, LibFunc_free},
    {Intrinsic::kit_reduce_num_partials, LibFunc_kitpthr_reduce_num_partials},
    {Intrinsic::kit_runtime_finalize, LibFunc_kitpthr_finalize},
    {Intrinsic::kit_runtime_initialize, LibFunc_kitpthr_initialize},
};

/// Kitsune runtime functions for the qthreads tapir target.
static const KitRTFuncMap kitQthreadsFuncs = {
    {Intrinsic::kit_cpu_threads_launch, LibFunc_kitqthr_launch},
    // There may be some benefit to using the memory allocation functions
    // provided by qthreads. Those use memory pools and it is not yet clear if
    // that is something we should consider using.
    {Intrinsic::kit_mobile_alloc, LibFunc_malloc},
    {Intrinsic::kit_mobile_free, LibFunc_free},
    {Intrinsic::kit_reduce_num_partials, LibFunc_kitqthr_reduce_num_partials},
    {Intrinsic::kit_runtime_finalize, LibFunc_kitqthr_finalize},
    {Intrinsic::kit_runtime_initialize, LibFunc_kitqthr_initialize},
};

/// Runtime library function maps for tapir targets that have a corresponding
/// kitsune runtime.
static const SmallDenseMap<TTID, KitRTFuncMap> kitTTFuncs = {
    {TTID::Cuda, kitCudaFuncs},         {TTID::Hip, kitHipFuncs},
    {TTID::OpenCilk, kitOpenCilkFuncs}, {TTID::OpenMP, kitOpenMPFuncs},
    {TTID::Pthreads, kitPthreadsFuncs}, {TTID::Qthreads, kitQthreadsFuncs},
};

/// When lowering the kitsune intrinsics, some arguments may need to be dropped
/// or reordered. The values in this map are the order in which the source
/// operands should appear in the call to the runtime function. For instance,
/// if the value is {2, 1, 3}, it indicates that the first argument to the
/// intrinsic is dropped, while the second and third arguments are swapped.
/// For e.g., if Intrinsic::example were to be lowered to LibFunc_example with
/// the argument map {2, 1, 3}, then this call:
///
///     call void llvm.example(i32 2, ptr writeonly a, ptr readonly b, i64 c)
///
/// would be lowered to
///
///     call void example(ptr readonly b, ptr writeonly a, i64 c)
///
/// Note that the attributes on the call arguments have been preserved. In fact,
/// this is the primary motivation for having this map, since, without it, these
/// attributes would likely be lost.
///
/// NOTE: It is not clear if there is any advantage to preserving these
/// attributes since this pass runs as part of codegen, at which point the
/// optimization pipeline has already run. Even so, it is probably a good idea
/// to preserve these attributes if only to avoid any unnecessary surprises if
/// this part of the code were ever moved elsewhere, or if a (very late)
/// optimization pass were added after this pass.
///
/// If an intrinsic does not contain an entry in this map, a custom lowering
/// function must be provided for it. If the arguments to an intrinsic must be
/// modified before passing them to the runtime function, the lowering *MUST* be
/// handled with a custom lowering function.
static const KitRTFuncArgMap kitRTArgMap = {
    {Intrinsic::kit_async_cpu_threads_launch, {1, 2, 3, 4, 5}},
    {Intrinsic::kit_cpu_threads_launch, {1, 2, 3, 4, 5}},
    {Intrinsic::kit_cpu_threads_sync, {1}},
    {Intrinsic::kit_async_gpu_memcpy_dtoh, {1, 2, 3, 4}},
    {Intrinsic::kit_async_gpu_memcpy_htod, {1, 2, 3, 4}},
    {Intrinsic::kit_async_gpu_prefetch_dtoh, {1, 3}},
    {Intrinsic::kit_async_gpu_prefetch_htod, {1, 3}},
    {Intrinsic::kit_gpu_memcpy_dtoh, {1, 2, 3}},
    {Intrinsic::kit_gpu_memcpy_htod, {1, 2, 3}},
    {Intrinsic::kit_gpu_stream_sync, {1}},
    {Intrinsic::kit_gpu_stream_new, {}},
    {Intrinsic::kit_gpu_symbol_address, {1, 2}},
    {Intrinsic::kit_gpu_symbol_memcpy_dtoh, {2, 1, 3}},
    {Intrinsic::kit_gpu_symbol_memcpy_htod, {2, 1, 3}},
    {Intrinsic::kit_reduce_num_partials, {1}},
    {Intrinsic::kit_runtime_finalize, {}},
    {Intrinsic::kit_runtime_initialize, {}},
    {Intrinsic::kit_runtime_set_fixed_tpb, {1}},
    {Intrinsic::kit_runtime_set_kernel_launch_refinement, {1}},
    {Intrinsic::kit_runtime_set_max_tpb, {1}},
    {Intrinsic::kit_runtime_set_verbose, {}},
    {Intrinsic::kit_runtime_set_xnack, {}},
    {Intrinsic::kit_runtime_set_y_axis_kernel_launch, {}},
};

/// Main implementation class to lower Kitsune intrinsics.
class LowerKitIntrinsics {
private:
  const TargetLibraryInfo &tli;

private:
  /// All runtime intrinsics take the TTID as the first argument. Parse this
  /// into a TTID enum.
  TTID getTTID(Value *v) const {
    assert(isa<Constant>(v) && "TTID must be a constant");
    assert(fromConstant<TTID>(*cast<Constant>(v)) && "Not a valid TTID");

    return *fromConstant<TTID>(*cast<Constant>(v));
  }

  FunctionCallee getOrInsertLibFunc(Module &m, LibFunc libFunc) {
    FunctionCallee f = llvm::getOrInsertLibFunc(&m, tli, libFunc);
    inferNonMandatoryLibFuncAttrs(*cast<Function>(f.getCallee()), tli);
    return f;
  }

  FunctionCallee getOrInsertLibFunc(Module &m, Intrinsic::ID id) {
    assert(kitFuncs.find(id) != kitFuncs.end() &&
           "getRuntimeFunc: No kitsune library function for intrinsic");
    return getOrInsertLibFunc(m, kitFuncs.at(id));
  }

  FunctionCallee getOrInsertLibFunc(Module &m, TTID tt, Intrinsic::ID id) {
    assert(kitTTFuncs.find(tt) != kitTTFuncs.end() &&
           "getRuntimeFunc: Invalid tapir target for intrinsic");
    const KitRTFuncMap &funcs = kitTTFuncs.at(tt);

    assert(funcs.find(id) != funcs.end() &&
           "getRuntimeFunc: No kitsune library function for tapir target");
    return getOrInsertLibFunc(m, funcs.at(id));
  }

  FunctionCallee getMobileAllocFunc(Module &m, CallInst &call) {
    Intrinsic::ID id = call.getIntrinsicID();
    TTID tt = *getTTIDFromKitIntrCall(call);

    switch (tt) {
    case TTID::Nolo:
      return nullptr;
    case TTID::Serial:
      return getOrInsertLibFunc(m, LibFunc_malloc);
    case TTID::Cuda:
    case TTID::Hip:
    case TTID::OpenCilk:
    case TTID::OpenMP:
    case TTID::Pthreads:
    case TTID::Qthreads:
      return getOrInsertLibFunc(m, tt, id);
    case TTID::Custom:
      // TODO: A custom tapir target may require a custom memory allocator.
      // Currently, there is no way to have the plugin specify a memory
      // allocator to use, so just default to using libc's malloc.
      return getOrInsertLibFunc(m, LibFunc_malloc);
    case TTID::Lambda:
    case TTID::OMPTask:
    case TTID::Realm:
      // These tapir targets are not fully supported yet, but add them to this
      // switch to ensure that a warning is emitted when a new tapir target is
      // added.
      llvm_unreachable("getMobileAllocFunc: TTID not handled");
    }
  }

  FunctionCallee getMobileFreeFunc(Module &m, CallInst &call) {
    Intrinsic::ID id = call.getIntrinsicID();
    TTID tt = *getTTIDFromKitIntrCall(call);

    switch (tt) {
    case TTID::Nolo:
      return nullptr;
    case TTID::Serial:
      return getOrInsertLibFunc(m, LibFunc_free);
    case TTID::Cuda:
    case TTID::Hip:
    case TTID::OpenCilk:
    case TTID::OpenMP:
    case TTID::Pthreads:
    case TTID::Qthreads:
      return getOrInsertLibFunc(m, tt, id);
    case TTID::Custom:
      // TODO: A custom tapir target may require a custom memory deallocator.
      // Currently, there is no way to have the plugin specify a memory
      // deallocator to use, so just default to using libc's free.
      return getOrInsertLibFunc(m, LibFunc_free);
    case TTID::Lambda:
    case TTID::OMPTask:
    case TTID::Realm:
      // These tapir targets are not fully supported yet, but add them to this
      // switch to ensure that a warning is emitted when a new tapir target is
      // added.
      llvm_unreachable("getMobileFreeFunc: TTID not handled");
    }
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
      return getOrInsertLibFunc(m, LibFunc_kitrt_mobile_init_bool);
    else if (isInt8(init))
      return getOrInsertLibFunc(m, LibFunc_kitrt_mobile_init_i8);
    else if (isInt16(init))
      return getOrInsertLibFunc(m, LibFunc_kitrt_mobile_init_i16);
    else if (isInt32(init))
      return getOrInsertLibFunc(m, LibFunc_kitrt_mobile_init_i32);
    else if (isInt64(init))
      return getOrInsertLibFunc(m, LibFunc_kitrt_mobile_init_i64);
    else if (isFloat(init))
      return getOrInsertLibFunc(m, LibFunc_kitrt_mobile_init_float);
    else if (isDouble(init))
      return getOrInsertLibFunc(m, LibFunc_kitrt_mobile_init_double);
    else if (isPointer(init))
      return getOrInsertLibFunc(m, LibFunc_kitrt_mobile_init_from);
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

    case Intrinsic::kit_runtime_set_verbose:
      // Intrinsics with runtime functions that are independent of a tapir
      // target.
      return getOrInsertLibFunc(m, id);

    case Intrinsic::kit_runtime_set_xnack:
      // Intrinsics that are exclusive to the hip tapir target
      return getOrInsertLibFunc(m, TTID::Hip, id);

    default:
      // Intrinsics with runtime functions dependent on the tapir target.
      return getOrInsertLibFunc(m, getTTID(call.getArgOperand(0)), id);
    }
  }

  /// Return a new attribute list which is exactly the same as the given
  /// attribute list \ref attrs except that the attributes at index \ref src of
  /// \ref call's attribute list are added to index \ref dst of \ref attrs. The
  /// newly created attribute list is returned.
  AttributeList addAttrsFrom(AttributeList attrs, unsigned dst,
                             const CallInst &call, unsigned src) {
    LLVMContext &ctx = call.getContext();
    AttributeList callAttrs = call.getAttributes();
    for (const Attribute &attr : callAttrs.getAttributes(src))
      attrs = attrs.addAttributeAtIndex(ctx, dst, attr);
    return attrs;
  }

  /// Return a new attribute list which is exactly the same as the given
  /// attribute list \ref attrs except that the attributes at index \ref src of
  /// \ref call's attribute list are added to index \ref src of \ref attrs. The
  /// newly created attribute list is returned.
  AttributeList addAttrsFrom(AttributeList attrs, const CallInst &call,
                             unsigned src) {
    return addAttrsFrom(attrs, src, call, src);
  }

  /// Create a new attribute list that will eventually be applied to the
  /// replacement of the given call instruction. If \ref argMap[i] = j, the
  /// attributes from the j'th argument of \ref call will be added to index i
  /// of the new attribute list that is returned.
  AttributeList createNewAttrList(const CallInst &call,
                                  ArrayRef<unsigned> argMap) {
    AttributeList attrs;
    attrs = addAttrsFrom(attrs, call, AttributeList::FunctionIndex);
    attrs = addAttrsFrom(attrs, call, AttributeList::ReturnIndex);
    for (size_t i = 0; i < argMap.size(); ++i) {
      unsigned isrc = AttributeList::FirstArgIndex + argMap[i];
      unsigned idst = AttributeList::FirstArgIndex + i;
      attrs = addAttrsFrom(attrs, idst, call, isrc);
    }
    return attrs;
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
    newCall->setAttributes(createNewAttrList(call, {1}));

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
    newCall->setAttributes(createNewAttrList(call, {1}));

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
    newCall->setAttributes(createNewAttrList(call, {1, 2, 3}));

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

    BasicBlock &bbEntry = call.getParent()->getParent()->getEntryBlock();

    SmallVector<Value *, 8> kernelArgs = getKernelArgumentsFromLaunch(call);
    ArrayType *arrTy = ArrayType::get(ptrTy, kernelArgs.size());
    AllocaInst *argArray = new AllocaInst(arrTy, 0, "", bbEntry.begin());
    for (size_t i = 0; i < kernelArgs.size(); ++i) {
      Constant *ci = ConstantInt::get(i64, i);
      Value *indices[] = {c0, ci};
      Value *kernelArg = kernelArgs[i];
      Type *argTy = kernelArg->getType();

      AllocaInst *slot = new AllocaInst(argTy, 0, "", argArray->getIterator());
      (void)new StoreInst(kernelArg, slot, call.getIterator());
      GetElementPtrInst *argOffset = GetElementPtrInst::CreateInBounds(
          arrTy, argArray, indices, "", call.getIterator());
      (void)new StoreInst(slot, argOffset, call.getIterator());
    }

    // The attributes can mostly be copied over to the new call. However, the
    // 3rd argument in the runtime call will be the packed argument array, so
    // those attributes cannot be copied over from the intrinsic call.
    unsigned attr0 = AttributeList::FirstArgIndex;
    AttributeList attrs;
    attrs = addAttrsFrom(attrs, call, AttributeList::FunctionIndex);
    attrs = addAttrsFrom(attrs, call, AttributeList::ReturnIndex);
    attrs = addAttrsFrom(attrs, attr0 + 0, call, attr0 + 1);
    attrs = addAttrsFrom(attrs, attr0 + 1, call, attr0 + 2);
    attrs = addAttrsFrom(attrs, attr0 + 3, call, attr0 + 3);
    attrs = addAttrsFrom(attrs, attr0 + 4, call, attr0 + 4);
    attrs = addAttrsFrom(attrs, attr0 + 5, call, attr0 + 5);
    attrs = addAttrsFrom(attrs, attr0 + 6, call, attr0 + 6);
    attrs = addAttrsFrom(attrs, attr0 + 7, call, attr0 + 7);
    attrs = addAttrsFrom(attrs, attr0 + 8, call, attr0 + 8);
    attrs = attrs.addAttributeAtIndex(ctx, attr0 + 2, Attribute::NonNull);

    FunctionCallee rtFunc = getRuntimeFunc(call);
    Value *args[] = {
        call.getArgOperand(1), // fatbin
        call.getArgOperand(2), // kernel name
        argArray,              // kernel arguments
        call.getArgOperand(3), // trip count (x)
        call.getArgOperand(4), // trip count (y)
        call.getArgOperand(5), // trip count (z)
        call.getArgOperand(6), // threads per block
        call.getArgOperand(7), // instruction mix
        call.getArgOperand(8), // stream
    };
    CallInst *newCall = createNewCallFor(call, rtFunc, args);

    newCall->setAttributes(attrs);
    call.replaceAllUsesWith(newCall);
    call.eraseFromParent();

    return true;
  }

  /// Replace the Kitsune intrinsic called in the given instruction with an
  /// appropriate runtime function to be called with the given arguments.
  /// Always returns true.
  bool lowerIntrinsicDefault(CallInst &call) {
    assert((kitRTArgMap.find(call.getIntrinsicID()) != kitRTArgMap.end()) &&
           "Intrinsic supports default lowering");

    ArrayRef argMap = kitRTArgMap.at(call.getIntrinsicID());
    FunctionCallee f = getRuntimeFunc(call);
    SmallVector<Value *, 4> args;
    for (unsigned argNo : argMap)
      args.push_back(call.getArgOperand(argNo));

    CallInst *newCall = createNewCallFor(call, f, args);
    newCall->setAttributes(createNewAttrList(call, argMap));

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
    bool changed = false;

    switch (call.getIntrinsicID()) {
    case Intrinsic::kit_mobile_alloc:
      changed |= lowerMobileAlloc(call);
      break;

    case Intrinsic::kit_mobile_free:
      changed |= lowerMobileFree(call);
      break;

    case Intrinsic::kit_mobile_init:
      changed |= lowerMobileInit(call);
      break;

    case Intrinsic::kit_async_gpu_kernel_launch:
      changed |= lowerLaunchKernel(call);
      break;

    case Intrinsic::kit_runtime_set_verbose:
    case Intrinsic::kit_runtime_set_xnack:
    case Intrinsic::kit_runtime_set_y_axis_kernel_launch:
      // The first argument is the TTID. The second is a flag. If the flag is
      // false, the corresponding runtime function should not be called.
      if (isZero(call.getArgOperand(1)))
        call.eraseFromParent();
      else
        lowerIntrinsicDefault(call);
      changed |= true;
      break;

    default:
      changed |= lowerIntrinsicDefault(call);
      break;
    }

    return changed;
  }

public:
  LowerKitIntrinsics(TargetLibraryInfo &tli) : tli(tli) {}

  bool run(Function &f) {
    bool changed = false;

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

  void getAnalysisUsage(AnalysisUsage &au) const override {
    au.addRequired<TargetLibraryInfoWrapperPass>();
  }

  bool runOnFunction(Function &f) override {
    TargetLibraryInfo &tli =
        getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(f);

    return LowerKitIntrinsics(tli).run(f);
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
  TargetLibraryInfo &tli = am.getResult<TargetLibraryAnalysis>(f);

  // If any kitsune intrinsics were replaced, the call graph will have changed,
  // but other analyses will not have been invalidated.
  bool changed = LowerKitIntrinsics(tli).run(f);
  if (changed) {
    PreservedAnalyses pa;
    pa.preserve<FunctionAnalysisManagerCGSCCProxy>();
    pa.preserveSet<AllAnalysesOn<Function>>();
    return pa;
  } else {
    return PreservedAnalyses::all();
  }
}
