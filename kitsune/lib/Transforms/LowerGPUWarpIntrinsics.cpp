//===- LowerGPUWarpIntrinsics.cpp - Lower GPU warp intrinsics -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune-specific GPU warp intrinsics.
//
//===----------------------------------------------------------------------===//

#include "LowerGPUIntrinsicsImpl.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/IRBuilderUtils.h"
#include "kitsune/Core/IntrinsicUtils.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/TTID.h"
#include "kitsune/Core/TTOptions.h"
#include "kitsune/Core/TypeUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/TargetParser.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

namespace {

// Base class to replace a warp intrinsic.
class LowerWarpIntrinsicBase {
private:
  Value *replaceImpl(CallInst *call) {
    switch (*getTTIDFromKitIntrCall(*call)) {
    case TTID::Cuda: return replaceIntrCuda(call);
    case TTID::Hip: return replaceIntrHip(call);
    default: llvm_unreachable("LowerWarpIntrinsicBase: TTID not handled!");
    }
  }

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
    f->setMemoryEffects(MemoryEffects::none());

    return f;
  }

  template <typename... T>
  std::string makeImplName(TTID tt, StringRef base, T &&...rest) {
    return join_items(".", "__kit", toString(tt), "warp", base, rest...);
  }

protected:
  virtual Value *replaceIntrCuda(CallInst *call) = 0;
  virtual Value *replaceIntrHip(CallInst *call) = 0;

public:
  bool run(CallInst *call) {
    BasicBlock::iterator pos = call->getIterator();
    ReplaceInstWithValue(pos, replaceImpl(call));
    return true;
  }
};

} // namespace

//==---------------------------- LowerWarpSize -----------------------------==//

namespace {

class LowerWarpSize : public LowerWarpIntrinsicBase {
protected:
  const TTOptions &tto;

protected:
  virtual Value *replaceIntrCuda(CallInst *call) override;
  virtual Value *replaceIntrHip(CallInst *call) override;

public:
  LowerWarpSize(const TTOptions &tto) : tto(tto) {}
};

} // namespace

// On NVIDIA GPU's, the warp size is always 32.
Value *LowerWarpSize::replaceIntrCuda(CallInst *call) {
  return toConstant(32U, call->getContext());
}

Value *LowerWarpSize::replaceIntrHip(CallInst *call) {
  auto getWavefrontSize = [](StringRef arch) -> unsigned {
    if (AMDGPU::GPUKind kind = AMDGPU::parseArchAMDGCN(arch)) {
      const unsigned archAttrs = AMDGPU::getArchAttrAMDGCN(kind);
      const bool hasWave32 = (archAttrs & AMDGPU::FEATURE_WAVE32);
      return hasWave32 ? 32 : 64;
    }
    return 0;
  };

  // Dealing with the warp size on AMDGPU is tricky. Some devices only support a
  // warp size of 32, others only support 64. But a few support both. To
  // determine which to use, we first check the hip features set in the
  // compiler.
  //
  //   - If either "+wavefrontsize32", or "+wavefrontsize64" are present in
  //     target features for the architecture, set the wavefront depending on
  //     which of the two is present. We check the features in the hip target
  //     features set in the tapir target options object. If the intrinsic is
  //     called in a function generated during lowering of a different
  //     intrinsic, the target features may not be set on the function.
  //
  //   - Otherwise, check the device architecture set in the hip tapir target
  //     options and use the default wavefront size for that architecture.
  //
  // At this point, if we have been unable to determine a wavefrontsize for
  // whatever reason, raise an error. We do not revert to a default because
  // there is no guarantee that the chosen default will work across devices.
  StringRef features = tto.getHipTargetFeatures();
  bool hasFeature32 = features.contains("+wavefrontsize32");
  bool hasFeature64 = features.contains("+wavefrontsize64");
  unsigned wavefrontSize = 0;
  if (hasFeature32 && !hasFeature64)
    wavefrontSize = 32;
  else if (hasFeature64 && !hasFeature32)
    wavefrontSize = 64;
  else if (unsigned w = getWavefrontSize(tto.getHipArch()))
    wavefrontSize = w;
  else
    llvm_unreachable("Could not determine wavefront size");

  return toConstant(wavefrontSize, call->getContext());
}

//==-------------------------- LowerWarpIdOrLane ---------------------------==//

namespace {

class LowerWarpIdOrLane : public LowerWarpIntrinsicBase {
protected:
  // This will be either "id" or "lane"
  StringRef baseName;

  // The operator to calculate the final result from the offset and the warp
  // size. This must be either UDiv or URem.
  Instruction::BinaryOps op;

protected:
  Value *genOffset(IRBuilder<> &builder, Value *tt);
  Function *getOrInsertImplFunc(Module &m, CallInst &call);
  Value *replaceIntr(CallInst *call);

protected:
  virtual Value *replaceIntrCuda(CallInst *call) override;
  virtual Value *replaceIntrHip(CallInst *call) override;

public:
  LowerWarpIdOrLane(StringRef baseName, Instruction::BinaryOps op)
      : baseName(baseName), op(op) {}
};

} // namespace

// Calculate the offset based on the threadIdx and blockIdx in all three
// dimensions. This is the general calculation:
//
//   tid = threadIdx.x
//           + threadIdx.y * blockDim.x
//           + threadIdx.z * blockDim.x * blockDim.y
//
// This will be correct even if some dimensions are not used since the threadIdx
// and blockDim values in those dimensions will be 0. This is not wasteful
// because the functions is guaranteed to be called with compile-time constants.
// Between inlining and the standard function simplification passes that will
// definitely be run after this has been generated, any unnecessary calculations
// will be optimized away.
Value *LowerWarpIdOrLane::genOffset(IRBuilder<> &builder, Value *tt) {
  LLVMContext &ctx = builder.getContext();
  Type *i32 = Type::getInt32Ty(ctx);
  Constant *zero = ConstantInt::get(i32, 0, /*isSigned=*/false);
  Constant *one = ConstantInt::get(i32, 1, /*isSigned=*/false);
  Constant *two = ConstantInt::get(i32, 2, /*isSigned=*/false);

  // Calculate the offset in the Z-dimension. This is the calculation:
  //
  //     offz = threadIdx.z * blockDim.x * blockDim.y
  //
  Value *bszx = builder.CreateIntrinsic(Intrinsic::kit_gpu_block_size_x, {tt},
                                        /*FMFSource=*/{}, "bszx");
  Value *bszy = builder.CreateIntrinsic(Intrinsic::kit_gpu_block_size_y, {tt},
                                        /*FMFSource=*/{}, "bszy");
  Value *bszxy = builder.CreateMul(bszx, bszy, "bszxy");
  Value *tidz = builder.CreateIntrinsic(Intrinsic::kit_gpu_thread_id_z, {tt},
                                        /*FMFSource=*/{}, "tidz");
  Value *offz = builder.CreateMul(tidz, bszxy, "offz");

  // Calculate the offset in the Y-dimension. This is the calculation
  //
  //     offy = threadIdx.y * blockDim.x
  //
  Value *tidy = builder.CreateIntrinsic(Intrinsic::kit_gpu_thread_id_y, {tt},
                                        /*FMFSource=*/{}, "tidy");
  Value *offy = builder.CreateMul(tidy, bszx, "offy");

  // The offset in the X-dimension is just the thread index in that dimension.
  //
  //     offx = threadIdx.x
  //
  // This essentially sets the final offset to offx i.e.
  //
  //     off = offx
  //
  Value *off = builder.CreateIntrinsic(Intrinsic::kit_gpu_thread_id_x, {tt},
                                       /*FMFSource=*/{}, "x");

  Function *f = getFunction(builder);
  Value *dims = f->getArg(0);

  // Conditionally add the offset in the Y-dimension. This gives the optimizer
  // a chance to eliminate the computation since the dimension argument is
  // known at compile-time.
  //
  //     off += (dims > 1) ? offy : 0
  //
  Value *hasy = builder.CreateICmpUGT(dims, one, "hasy");
  Value *y = builder.CreateSelect(hasy, offy, zero, "y");
  off = builder.CreateAdd(off, y, "offxy");

  // Conditionally add the offset in the Z-dimension. This gives the optimizer
  // a chance to eliminate the computation since the dimension argument is
  // known at compile-time.
  //
  //     off += (dims > 2) ? offz : 0;
  //
  Value *hasz = builder.CreateICmpUGT(dims, two, "hasz");
  Value *z = builder.CreateSelect(hasz, offz, zero, "z");
  off = builder.CreateAdd(off, z, "offxyz");

  return off;
}

// The implementation function returns a 32-bit integer since the warp index and
// lane are guaranteed to be 32 bits, and takes a 32-bit integer as the sole
// argument. This argument is a hint about the number of dimensions in the
// computation.
Function *LowerWarpIdOrLane::getOrInsertImplFunc(Module &m, CallInst &call) {
  // The name of the implementation function of these intrinsics only depends on
  // the TTID. We pass the number of dimensions as an argument to it. Since that
  // argument is guaranteed to be a compile-time constant, the optimizer will
  // get rid of any unnecessary calculations. As a result, there is no need to
  // specialize this for the various dimensions.
  std::string implName = makeImplName(*getTTIDFromKitIntrCall(call), baseName);
  if (Function *repl = m.getFunction(implName))
    return repl;

  LLVMContext &ctx = m.getContext();
  Type *i32 = Type::getInt32Ty(ctx);
  Function *f = this->addImplFunc(m, implName, i32, i32);
  f->getArg(0)->setName("dims");

  Value *tt = call.getArgOperand(0);
  BasicBlock *entry = BasicBlock::Create(ctx, "", f);
  IRBuilder<> builder(entry);
  Value *offset = genOffset(builder, tt);
  Value *warpSz = builder.CreateIntrinsic(Intrinsic::kit_gpu_warp_size, {tt});
  Value *res = builder.CreateBinOp(op, offset, warpSz);
  builder.CreateRet(res);

  return f;
}

Value *LowerWarpIdOrLane::replaceIntr(CallInst *call) {
  Function *replF = getOrInsertImplFunc(*call->getModule(), *call);
  FunctionType *replTy = replF->getFunctionType();

  Value *dims = call->getArgOperand(1);
  InsertPosition insertPt = call->getIterator();
  CallInst *newCall = CallInst::Create(replTy, replF, {dims}, "", insertPt);

  return newCall;
}

Value *LowerWarpIdOrLane::replaceIntrCuda(CallInst *call) {
  return replaceIntr(call);
}

Value *LowerWarpIdOrLane::replaceIntrHip(CallInst *call) {
  return replaceIntr(call);
}

//==--------------------------- LowerWarpShuffle ---------------------------==//

namespace {

class LowerWarpShuffleDownSync : public LowerWarpIntrinsicBase {
protected:
  std::string getCoreImplName(TTID tt);
  Function *getOrInsertLaneIdImplHip(Module &m);
  Function *getOrInsertCoreImplHip(Module &m);
  Function *getOrInsertCoreImplCuda(Module &m);
  Function *getOrInsertPiecewiseImpl(Module &m, TTID tt, Function *coreImpl,
                                     Type *ty);
  Value *genPiecewise32(IRBuilder<> &builder, TTID tt, Function *coreImpl,
                        Value *val, Value *offset);
  Value *genPiecewise64(IRBuilder<> &builder, TTID tt, Function *coreImpl,
                        Value *val, Value *offset);
  Value *replaceIntr(CallInst *call, TTID tt, Function *coreImpl);

protected:
  virtual Value *replaceIntrCuda(CallInst *call) override;
  virtual Value *replaceIntrHip(CallInst *call) override;
};

} // namespace

std::string LowerWarpShuffleDownSync::getCoreImplName(TTID tt) {
  return makeImplName(tt, "shfl.down.sync.core");
}

Function *LowerWarpShuffleDownSync::getOrInsertCoreImplCuda(Module &m) {
  std::string fname = getCoreImplName(TTID::Cuda);
  if (Function *f = m.getFunction(fname))
    return f;

  LLVMContext &ctx = m.getContext();
  Type *i32 = Type::getInt32Ty(ctx);
  Function *f = addImplFunc(m, fname, i32, i32, i32);
  f->getArg(0)->setName("val");
  f->getArg(1)->setName("offset");

  BasicBlock *entry = BasicBlock::Create(ctx, "", f);
  IRBuilder<> builder(entry);

  // This implementation calls the @llvm.nvvm.shfl.down.sync intrinsic. The
  // default NVIDIA implementation is something like this:
  //
  //     c = ((warpSize - width) << 8) | 0x1f;
  //     __nvvm_shfl_down_sync(mask, var, offset, c)
  //
  // Here mask is a bitmask indicating which threads are participate in the
  // warp shuffle. Since we only use this when lowering reductions, and we do
  // not support conditional reductions, the mask is always 0xFFFFFFFFU. By
  // default, even in cuda, width == warpSize. Presumably the reason to have
  // this be different is if only some threads in a warp were participating in
  // the shuffle. As we have already stated, we require all threads to
  // participate, so we assume that `c` is 0x1f.
  //
  // The implementation of this function then simply becomes a call to the
  // intrinsic.
  Value *val = f->getArg(0);
  Value *offset = f->getArg(1);
  Value *maskThrds = toConstant(0xffffffffU, ctx);
  Value *maskLanes = toConstant(0x1fU, ctx);
  Value *result = builder.CreateIntrinsic(Intrinsic::nvvm_shfl_sync_down_i32,
                                          {maskThrds, val, offset, maskLanes});
  builder.CreateRet(result);

  return f;
}

Function *LowerWarpShuffleDownSync::getOrInsertLaneIdImplHip(Module &m) {
  StringRef fname = "__kit.hip.lane.id";
  if (Function *f = m.getFunction(fname))
    return f;

  LLVMContext &ctx = m.getContext();
  Type *i32 = Type::getInt32Ty(ctx);

  Constant *ctt = toConstant(TTID::Hip, ctx);
  Constant *c32 = toConstant(32U, ctx);
  Constant *neg1 = toConstant(-1, ctx);
  Constant *zero = toConstant(0, ctx);

  Function *f = addImplFunc(m, fname, i32);

  BasicBlock *bbEntry = BasicBlock::Create(ctx, "entry", f);
  BasicBlock *bb64 = BasicBlock::Create(ctx, "64", f);
  BasicBlock *bbExit = BasicBlock::Create(ctx, "exit", f);

  IRBuilder<> builder(ctx);

  builder.SetInsertPoint(bbEntry);
  Value *warpSize =
      builder.CreateIntrinsic(Intrinsic::kit_gpu_warp_size, {ctt});
  Value *lane32 = builder.CreateIntrinsic(
      Intrinsic::amdgcn_mbcnt_lo, {neg1, zero}, /*FMFSource=*/{}, "l32");
  Value *is32 = builder.CreateICmpEQ(warpSize, c32, "is32");
  builder.CreateCondBr(is32, bbExit, bb64);

  builder.SetInsertPoint(bb64);
  Value *lane64 =
      builder.CreateIntrinsic(Intrinsic::amdgcn_mbcnt_hi, {neg1, lane32});
  builder.CreateBr(bbExit);

  builder.SetInsertPoint(bbExit);
  PHINode *result = builder.CreatePHI(i32, /*reserved=*/32, "res");
  result->addIncoming(lane32, bbEntry);
  result->addIncoming(lane64, bb64);
  builder.CreateRet(result);

  return f;
}

Function *LowerWarpShuffleDownSync::getOrInsertCoreImplHip(Module &m) {
  std::string fname = getCoreImplName(TTID::Hip);
  if (Function *f = m.getFunction(fname))
    return f;

  LLVMContext &ctx = m.getContext();
  Type *i32 = Type::getInt32Ty(ctx);
  Function *f = addImplFunc(m, fname, i32, i32, i32);

  BasicBlock *entry = BasicBlock::Create(ctx, "", f);
  IRBuilder<> builder(entry);

  Value *ctt = toConstant(TTID::Hip, ctx);
  Value *one = toConstant(1U, ctx);

  Value *val = f->getArg(0);
  Value *offset = f->getArg(1);

  // This implementation does not perform any error checking. It assumes that,
  // when the offset is added to the current lane, the resulting lane is valid.
  // The basic implementation here is:
  //
  //     ngbr = (lane_id() & (width - 1)) + offset.
  //
  // Here `width` is the warp size. It is not clear why we need to mask the lane
  // id, but the implementation in AMD's implementation does so.
  Function *laneIdFunc = getOrInsertLaneIdImplHip(m);
  Value *laneId = builder.CreateCall(laneIdFunc, /*args=*/{}, "id");
  Value *width = builder.CreateIntrinsic(Intrinsic::kit_gpu_warp_size, {ctt});
  Value *mask = builder.CreateSub(width, one, "mask");
  Value *lane = builder.CreateAnd(laneId, mask, "lane");
  Value *ngbr = builder.CreateAdd(lane, offset, "ngbr");
  Value *index = builder.CreateShl(ngbr, 2, "index");
  Value *result =
      builder.CreateIntrinsic(Intrinsic::amdgcn_ds_bpermute, {index, val});
  builder.CreateRet(result);

  return f;
}

Value *LowerWarpShuffleDownSync::genPiecewise32(IRBuilder<> &builder, TTID tt,
                                                Function *coreImpl, Value *val,
                                                Value *offset) {
  Type *ty = val->getType();

  LLVMContext &ctx = builder.getContext();
  Type *i32 = Type::getInt32Ty(ctx);

  Value *v32 = builder.CreateBitCast(val, i32, "v32");
  Value *r32 = builder.CreateCall(coreImpl, {v32, offset});
  Value *res = builder.CreateBitCast(r32, ty, "res");

  return res;
}

Value *LowerWarpShuffleDownSync::genPiecewise64(IRBuilder<> &builder, TTID tt,
                                                Function *coreImpl, Value *val,
                                                Value *offset) {
  LLVMContext &ctx = builder.getContext();
  Type *i64 = Type::getInt64Ty(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  Type *ty = val->getType();

  // The 64-bit value must be split into 2 32-bit pieces.
  Value *v64 = builder.CreateBitCast(val, i64, "v64");

  // Truncate so we are left with the lower 32-bits. The core implementation
  // will return a 32-bit result which is then zero-extended to get the lower
  // 32-bits of the result.
  Value *l32 = builder.CreateTrunc(v64, i32, "l32");
  Value *resL32 =
      builder.CreateCall(coreImpl, {l32, offset}, /*FMFSource=*/{}, "rl32");
  Value *resL64 = builder.CreateZExt(resL32, i64, "rl64");

  // Shift right by 32 bits and obtain the upper 32 bits of the result.
  // Zero-extend it to 64 bits and shift it left to get the final upper 32 bits.
  Value *u64 = builder.CreateLShr(v64, 32);
  Value *u32 = builder.CreateTrunc(u64, i32, "u32", /*nuw=*/true);
  Value *resU32 =
      builder.CreateCall(coreImpl, {u32, offset}, /*FMFSource=*/{}, "ru32");
  Value *resU64Tmp = builder.CreateZExt(resU32, i64);
  Value *resU64 =
      builder.CreateShl(resU64Tmp, 32, "ru64", /*nuw=*/false, /*nsw=*/true);

  // Bitwise OR the bits to get the final result, and cast it back to the
  // correct type.
  Value *res64 = builder.CreateOr(resU64, resL64, "res64", /*disjoint=*/true);
  Value *res = builder.CreateBitCast(res64, ty, "res");

  return res;
}

Function *LowerWarpShuffleDownSync::getOrInsertPiecewiseImpl(Module &m, TTID tt,
                                                             Function *coreImpl,
                                                             Type *ty) {
  std::string fname = makeImplName(tt, "shfl.down.sync", getShortName(ty));
  if (Function *f = m.getFunction(fname))
    return f;

  LLVMContext &ctx = m.getContext();
  Type *i32 = Type::getInt32Ty(ctx);

  Function *f = addImplFunc(m, fname, ty, ty, i32);
  Value *val = f->getArg(0);
  Value *offset = f->getArg(1);

  val->setName("val");
  offset->setName("offset");
  unsigned size = ty->getPrimitiveSizeInBits();

  BasicBlock *entry = BasicBlock::Create(ctx, "", f);
  IRBuilder<> builder(entry);
  Value *result = nullptr;
  if (size == 32)
    result = genPiecewise32(builder, tt, coreImpl, val, offset);
  else if (size == 64)
    result = genPiecewise64(builder, tt, coreImpl, val, offset);
  else
    llvm_unreachable("getPiecewiseImpl: Unsupported type size");
  builder.CreateRet(result);

  return f;
}

Value *LowerWarpShuffleDownSync::replaceIntr(CallInst *call, TTID tt,
                                             Function *coreImpl) {
  Module &m = *call->getModule();
  Value *val = call->getArgOperand(1);
  Value *offset = call->getArgOperand(2);
  Type *ty = val->getType();
  Function *impl = getOrInsertPiecewiseImpl(m, tt, coreImpl, ty);
  FunctionType *implTy = impl->getFunctionType();
  InsertPosition insertPt = call->getIterator();
  CallInst *newCall =
      CallInst::Create(implTy, impl, {val, offset}, "", insertPt);

  return newCall;
}

Value *LowerWarpShuffleDownSync::replaceIntrCuda(CallInst *call) {
  // There is a single core implementation for a given TTID, so generate it
  // right away.
  Function *coreImpl = getOrInsertCoreImplCuda(*call->getModule());
  return replaceIntr(call, TTID::Cuda, coreImpl);
}

Value *LowerWarpShuffleDownSync::replaceIntrHip(CallInst *call) {
  // Some core implementation functions are not specialized by type. We might
  // as well generate those early.
  getOrInsertLaneIdImplHip(*call->getModule());
  Function *coreImpl = getOrInsertCoreImplHip(*call->getModule());
  return replaceIntr(call, TTID::Hip, coreImpl);
}

//==------------------------------------------------------------------------==//

bool detail::LowerGPUIntrImpl::lowerWarpSizeIntr(CallInst *call) {
  return LowerWarpSize(tto).run(call);
}

bool detail::LowerGPUIntrImpl::lowerWarpIdIntr(CallInst *call) {
  return LowerWarpIdOrLane("id", Instruction::UDiv).run(call);
}

bool detail::LowerGPUIntrImpl::lowerWarpLaneIntr(CallInst *call) {
  return LowerWarpIdOrLane("lane", Instruction::URem).run(call);
}

bool detail::LowerGPUIntrImpl::lowerWarpShflDownSyncIntr(CallInst *call) {
  return LowerWarpShuffleDownSync().run(call);
}
