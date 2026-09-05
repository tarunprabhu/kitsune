//===- LowerGPUReduceIntrinsics.cpp - Lower GPU reduce intrinsics ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune-specific GPU reduction intrinsics.
//
//===----------------------------------------------------------------------===//

#include "LowerGPUIntrinsicsImpl.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/FuncAttrs.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/Reductions.h"
#include "kitsune/Core/TypeUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/IRBuilder.h"

#define DEBUG_TYPE "emb-lower-reduce-intrinsics"

using namespace llvm;

namespace {

class LowerBase {
protected:
  template <typename... Args>
  Function *addImplFunc(Module &m, StringRef name, Type *ret, Args &&...args) {
    Function *f = getOrInsertFunction(m, name, ret, args...);
    f->setLinkage(GlobalValue::LinkOnceODRLinkage);
    f->addFnAttr(Attribute::Convergent);
    f->addFnAttr(Attribute::MustProgress);
    f->addFnAttr(Attribute::NoFree);
    f->addFnAttr(Attribute::NoRecurse);
    f->addFnAttr(Attribute::NoUnwind);
    f->addFnAttr(Attribute::WillReturn);

    return f;
  }

  std::string makeImplName(StringRef base, unsigned dims,
                           const ReductionInfo &redxn) {
    TTID tt = redxn.tt;
    ReduceOp op = redxn.reduceOp;
    Type *type = redxn.getType();
    return join_items(".", "__kit.reduce", base, std::to_string(dims),
                      toString(tt), toString(op), getShortName(type));
  }

protected:
  virtual void lower(CallInst *call) = 0;

public:
  bool run(CallInst *call) {
    lower(call);
    call->eraseFromParent();
    return true;
  }
};

} // namespace

// --------------------- LowerWarpShuffleWithSharedMemory ---------------------

namespace {

// The contributions of each thread in a warp will be reduced to a warp-level
// contribution using the warp shuffle technique. A single thread in a each
// warp will write this contribution to a dedicated location in shared memory.
// The values will then be reduced to into a single value for the block. This
// value will be accumulated into the global result using an atomic operation.
class LowerWarpShuffleWithSharedMemory : public LowerBase {
protected:
  void lower(CallInst *call) override {
    llvm_unreachable("NOT YET IMPLEMENTED: ReduceWarpShuffleWithSharedMemory");
  }
};

}; // namespace

// ----------------------------- LowerSharedMemory -----------------------------

namespace {

// Every thread in a block will write its contribution to the final result into
// a dedicated location in shared memory. All of these elements will then be
// reduced into a single value for the block. This value will be accumulated
// into the global result using an atomic operation.
class LowerSharedMemory : public LowerBase {
protected:
  void lower(CallInst *call) override {
    llvm_unreachable("NOT YET IMPLEMENTED: ReduceSharedMemory");
  }
};

} // namespace

// ----------------------------- LowerWarpShuffle -----------------------------

namespace {

// The contributions of each thread in a warp will be reduced to a warp-level
// contribution using the warp shuffle technique. A single thread in a each
// warp will write accumulate this contribution directly into the global result
// using an atomic operation.
class LowerWarpShuffle : public LowerBase {
protected:
  Function *getOrInsertFinalReduceImpl(Module &m, unsigned dims,
                                       const ReductionInfo &redxn);
  Function *getOrInsertShuffleImpl(Module &m, unsigned dims,
                                   const ReductionInfo &redxn);
  void lower(CallInst *call) override;
};

} // namespace

// Generate the implementation that performs final reduction into the global
// result. This function takes two arguments - a pointer to the variable into
// which to perform the final reduction, and the value to accumulate into it.
// The code below is how this function might be written in high-level source:
//
//     void __kit.reduce.warp.shuffle.final.TT.OP.V(V *dest, V v) {
//       unsigned lane = __kit_gpu_warp_lane();
//       if (lane == 0)
//         __kit_gpu_reduce_direct(dest, OP, v);
//     }
//
// Note that only the first lane in each warp writes the final result. The
// implementation also results in the introduction of other intrinsic calls that
// have to be lowered.
Function *
LowerWarpShuffle::getOrInsertFinalReduceImpl(Module &m, unsigned dims,
                                             const ReductionInfo &redxn) {
  std::string implName = makeImplName("warp.shuffle.final", dims, redxn);
  if (Function *f = m.getFunction(implName))
    return f;

  LLVMContext &ctx = m.getContext();
  Type *voidTy = Type::getVoidTy(ctx);
  PointerType *ptr = PointerType::getUnqual(ctx);
  Type *i32 = Type::getInt32Ty(ctx);

  Constant *cdims = ConstantInt::get(i32, dims, /*isSigned=*/false);
  Constant *zero = ConstantInt::get(i32, 0, /*isSigned=*/false);

  Value *tt = redxn.getTTV();
  Value *op = redxn.getReduceOpV();
  Value *elemSize = redxn.getElemSizeV();
  Value *unit = redxn.getUnit();
  Value *reducer = redxn.getReducer();
  Type *ty = redxn.getType();

  Function *f = addImplFunc(m, implName, voidTy, ptr, ty);
  f->getArg(0)->setName("result");
  f->getArg(1)->setName("val");

  BasicBlock *bbEntry = BasicBlock::Create(ctx, "entry", f);
  BasicBlock *bbReduce = BasicBlock::Create(ctx, "reduce", f);
  BasicBlock *bbExit = BasicBlock::Create(ctx, "exit", f);

  Value *dest = f->getArg(0);
  Value *val = f->getArg(1);

  IRBuilder<> builder(ctx);

  builder.SetInsertPoint(bbEntry);
  Value *lane =
      builder.CreateIntrinsic(Intrinsic::kit_gpu_warp_lane, {tt, cdims});
  Value *is0 = builder.CreateICmpEQ(lane, zero);
  builder.CreateCondBr(is0, bbReduce, bbExit);

  builder.SetInsertPoint(bbReduce);
  SmallVector<Type *, 2> overloadTys = redxn.getOverloadTypes();
  SmallVector<Value *, 8> args = {tt, op, dest, elemSize, val, unit, reducer};
  args.append(redxn.getExtraArgs());
  builder.CreateIntrinsic(Intrinsic::kit_gpu_reduce_direct, overloadTys, args);
  builder.CreateBr(bbExit);

  builder.SetInsertPoint(bbExit);
  builder.CreateRetVoid();

  return f;
}

// Generate the core warp shuffle implementation. This function takes a single
// argument and returns a value of the same type. The argument is the value
// contribution of the calling thread to the final result. The code below is how
// this function might be written in high-level source:
//
//     V __kit.reduce.warp.shuffle.DIMS.TT.OP.V(V v) {
//       unsigned warpSize = __kit_gpu_warp_size();
//       for (unsigned offset = warpSize / 2; offset > 0; offset /= 2) {
//         V ngbr = __kit_gpu_warp_shuf_down_sync(val, offset);
//         __kit_reduce(&v, OP, ngbr);
//       }
//       return v;
//     }
//
// Here,
//
//     DIMS: The number of dimensions in the kernel that calls this intrinsic
//     OP:   The reduction operator
//     TT:   Name of the tapir target
//     V:    Type of the value being reduced
//
// Note that this introduces additional intrinsics that must themselves be
// lowered in a later step. Using the `shuf_down_sync` intrinsic, each thread
// obtains the value of the register containing `val` from a neighboring thread
// that is `offset` away from itself. Once control leaves the `offset` loop,
// the first thread in the warp will have a value that is the warp's
// contribution to the final reduction. Although every thread will return some
// calculated value, only the one returned by the call on this first thread
// should be accumulated into the final result.
Function *LowerWarpShuffle::getOrInsertShuffleImpl(Module &m, unsigned dims,
                                                   const ReductionInfo &redxn) {
  std::string implName = makeImplName("warp.shuffle", dims, redxn);
  if (Function *f = m.getFunction(implName))
    return f;

  LLVMContext &ctx = m.getContext();
  PointerType *ptr = PointerType::getUnqual(ctx);
  Type *i32 = Type::getInt32Ty(ctx);

  Value *tt = redxn.getTTV();
  Value *op = redxn.getReduceOpV();
  Value *size = redxn.getElemSizeV();
  Value *unit = redxn.getUnit();
  Value *reducer = redxn.getReducer();
  Type *ty = redxn.getType();
  Constant *zero = ConstantInt::get(i32, 0, /*isSigned=*/false);
  Constant *two = ConstantInt::get(i32, 2, /*isSigned=*/false);

  Function *f = addImplFunc(m, implName, ty, ty);
  f->getArg(0)->setName("val");

  BasicBlock *bbEntry = BasicBlock::Create(ctx, "entry", f);
  BasicBlock *bbLoop = BasicBlock::Create(ctx, "loop", f);
  BasicBlock *bbExit = BasicBlock::Create(ctx, "exit", f);

  IRBuilder<> builder(ctx);

  builder.SetInsertPoint(bbEntry);
  AllocaInst *res = builder.CreateAlloca(ty, /*ArraySize=*/nullptr, "res");
  builder.CreateStore(f->getArg(0), res);
  Value *warpSize = builder.CreateIntrinsic(Intrinsic::kit_gpu_warp_size, {tt});
  Value *off0 = builder.CreateUDiv(warpSize, two, "off0");
  builder.CreateBr(bbLoop);

  builder.SetInsertPoint(bbLoop);
  PHINode *off = builder.CreatePHI(i32, /*Reserved=*/2, "offset");
  off->addIncoming(off0, bbEntry);
  Value *val = builder.CreateLoad(ty, res);
  Value *newVal =
      builder.CreateIntrinsic(Intrinsic::kit_gpu_warp_shfl_down_sync, {ty, ty},
                              {tt, val, off}, /*FMFSource=*/{}, "v");

  Value *res0 = builder.CreateAddrSpaceCast(res, ptr);
  SmallVector<Type *, 2> overloadTys = redxn.getOverloadTypes();
  SmallVector<Value *, 8> args = {tt, op, res0, size, newVal, unit, reducer};
  args.append(redxn.getExtraArgs());
  builder.CreateIntrinsic(Intrinsic::kit_reduce_0, overloadTys, args);

  Value *newOff = builder.CreateUDiv(off, two, "offset.next");
  Value *is0 = builder.CreateICmpEQ(newOff, zero, "offset.cmp");
  off->addIncoming(newOff, bbLoop);
  builder.CreateCondBr(is0, bbExit, bbLoop);

  builder.SetInsertPoint(bbExit);
  Value *ret = builder.CreateLoad(ty, res);
  builder.CreateRet(ret);

  return f;
}

void LowerWarpShuffle::lower(CallInst *call) {
  auto sanityCheck = [](CallInst *call) -> void {
    assert(hasKernelAttr(*call->getFunction()) &&
           "Function containing warp shuffle intrinsic must be a kernel");
  };

  sanityCheck(call);

  const ReductionInfo redxn(call);
  Value *value = redxn.getValue();
  Value *dest = redxn.getDest();

  Module &m = *call->getModule();
  Function &f = *call->getFunction();
  unsigned dims = *getKernelAttr(f);

  IRBuilder<> builder(call);
  Function *implShuffle = getOrInsertShuffleImpl(m, dims, redxn);
  Function *implReduce = getOrInsertFinalReduceImpl(m, dims, redxn);

  Value *reduced = builder.CreateCall(implShuffle, {value});
  (void)builder.CreateCall(implReduce, {dest, reduced});
}

// -------------------------------- LowerDirect --------------------------------

namespace {

// Each thread will accumulate its contribution directly into the final result
// variable using an atomic operation.
class LowerDirect : public LowerBase {
protected:
  virtual void lower(CallInst *call) override;
};

} // namespace

void LowerDirect::lower(CallInst *call) {
  const ReductionInfo redxn(call);
  ReduceOp reduceOp = redxn.reduceOp;
  Value *result = redxn.getDest();
  Value *value = redxn.getValue();

  Module &m = *call->getModule();
  Function &f = *call->getFunction();

  std::optional<AtomicRMWInst::BinOp> atomicOp = getAtomicOp(reduceOp);
  if (!atomicOp)
    emitDiagnostic(f, DiagID::ErrNYI,
                   "Reduction operator not supported by AtomicRMWInst");

  Align align = getPointerAlignment(m);
  InsertPosition insertPt = call->getIterator();
  (void)new AtomicRMWInst(*atomicOp, result, value, align,
                          AtomicOrdering::Monotonic, SyncScope::System,
                          insertPt);
}

// -----------------------------------------------------------------------------

bool detail::LowerGPUIntrImpl::lowerReduceDirectIntr(CallInst *call) {
  return LowerDirect().run(call);
}

bool detail::LowerGPUIntrImpl::lowerReduceShmemIntr(CallInst *call) {
  return LowerSharedMemory().run(call);
}

bool detail::LowerGPUIntrImpl::lowerReduceWarpShflIntr(CallInst *call) {
  return LowerWarpShuffle().run(call);
}

bool detail::LowerGPUIntrImpl::lowerReduceWarpShflShmemIntr(CallInst *call) {
  return LowerWarpShuffleWithSharedMemory().run(call);
}
