//===- PrepareReductionLoopsGPU.cpp - Transform reduction loops for GPU ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transform tapir loops that perform reductions to a form that is suitable for
// parallel execution. This is the first in a two-step process. This pass only
// allocates "shadow" memory into which the GPU reduction will be performed,
// then sets the destination operand of the reduction intrinsics to this memory.
// The "shadow" memory is allocated in UVM by default, but may also be allocated
// directly in global memory.
//
//     int64_t r_sum = 0;
//     int32_t r_and = 0;
//     parallel_for (int i = 0; i < n; ++i) {
//         kit.reduce.0(&r_sum, sizeof(r_sum), i, 0, &f_sum);
//         kit.reduce.0(&r_and, sizeof(r_and), i, 1, &f_and);
//     }
//
// This pass will transform this into the following for parallel execution on a
// GPU.
//
//     int32_t *g_sum = allocateShadowMemory<int32_t>(0);
//     int64_t *g_and = allocateShadowMemory<int64_t>(0);
//     parallel_for (int i = 0; j < n; ++i) {
//         kit.reduce.0(&g_sum, sizeof(int32_t), i, 0, &f_sum);
//         kit.reduce.0(&g_and, sizeof(int64_t), i, 1, &f_and);
//     }
//     copyToFinalResult(&r_sum, g_sum, sizeof(int32_t));
//     copyToFinalResult(&r_and, g_and, sizeof(int64_t));
//     freeShadowMemory(g_sum);
//     freeShadowMemory(g_and);
//
//
//===----------------------------------------------------------------------===//

#include "PrepareReductionLoops.h"
#include "kitsune/Core/AddrSpace.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/GPUMemUtils.h"
#include "kitsune/Core/IRBuilderUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/Reductions.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "llvm/IR/IRBuilder.h"

#define DEBUG_TYPE "kit-prepare"

using namespace llvm;

namespace {

// Transform tapir reduction loops for parallel executions on GPU's.
class PrepareReductionLoopGPU {
private:
  Value *allocShadowMem(Loop &loop, GPUDynMemAllocKind dmem,
                        const ReductionInfo &redxn);
  void reduceIntoShadowMem(Loop &loop, Value *shadow,
                           const ReductionInfo &redxn);
  void copyShadowMemToDest(Loop &loop, GPUDynMemAllocKind dmem, Value *shadow,
                           Value *dest, const ReductionInfo &redxn);
  void freeShadowMem(Loop &loop, GPUDynMemAllocKind dmem, Value *shadow,
                     const ReductionInfo &redxn);

public:
  bool run(Loop &loop, const SmallVectorImpl<ReductionInfo> &reductions);
};

} // namespace

static cl::opt<GPUDynMemAllocKind>
    clShadow("tapir-gpu-reduce-shadow",
             cl::desc("Where to allocate the shadow reduction destination"),
             cl::init(GPUDynMemAllocKind::UVM), cl::value_desc("where"),
             cl::cat(cl::catKitClOpts),
             cl::values(clEnumValN(GPUDynMemAllocKind::Global, "global", ""),
                        clEnumValN(GPUDynMemAllocKind::UVM, "uvm", "")));

// Get the shadow memory kind to use. We always use UVM unless an override has
// been provided explicitly. Using UVM may result in page migrations between
// device and host, so it may not always be profitable. However, a profitability
// analysis requires at least a function analysis, so we do not do that here.
static GPUDynMemAllocKind getShadowMemoryKind() {
  if (clShadow.getNumOccurrences()) {
    if (clShadow == GPUDynMemAllocKind::Global)
      emitDiagnostic(DiagID::WarnGeneric,
                     "Using global memory for the shadow reduction variable is "
                     "known to be buggy and NOT recommended");
    return clShadow;
  }
  return GPUDynMemAllocKind::UVM;
}

// Allocate space, on the GPU, where the values being reduced will be
// accumulated. The allocated memory will be initialized with the value in the
// destination of the reduction.
Value *PrepareReductionLoopGPU::allocShadowMem(Loop &loop,
                                               GPUDynMemAllocKind dmem,
                                               const ReductionInfo &redxn) {
  auto allocShadowInGlobalMem = [](IRBuilder<> &builder, Value *size,
                                   const ReductionInfo &redxn) -> Value * {
    Value *tt = redxn.getTTV();
    Value *dest = redxn.getDest();
    Value *shadow = createGPUMalloc(builder, redxn.tt, size);
    builder.CreateIntrinsic(Intrinsic::kit_gpu_memcpy_htod,
                            {tt, shadow, dest, size});
    return shadow;
  };

  auto allocShadowInUVM = [](IRBuilder<> &builder, Value *size,
                             const ReductionInfo &redxn) -> Value * {
    LLVMContext &ctx = builder.getContext();
    PointerType *ptr = PointerType::getUnqual(ctx);

    Value *tt = redxn.getTTV();
    Value *dest = redxn.getDest();

    Value *b = builder.CreateIntrinsic(Intrinsic::kit_mobile_alloc, {tt, size});
    Value *shadow = builder.CreateAddrSpaceCast(b, ptr);
    builder.CreateMemCpyInline(shadow, MaybeAlign(), dest, MaybeAlign(), size);

    return shadow;
  };

  LLVM_DEBUG(dbgs() << "PrepareReductionGPU: Allocate shadow memory\n");

  LLVMContext &ctx = getContext(loop);
  Type *i64 = Type::getInt64Ty(ctx);

  Value *size = redxn.getElemSizeV();

  BasicBlock &bb = *loop.getLoopPreheader();
  IRBuilder<> builder(bb.getTerminator());

  Value *sz64 = builder.CreateIntCast(size, i64, /*isSigned=*/false);
  if (dmem == GPUDynMemAllocKind::Global)
    return allocShadowInGlobalMem(builder, sz64, redxn);
  else if (dmem == GPUDynMemAllocKind::UVM)
    return allocShadowInUVM(builder, sz64, redxn);
  else
    llvm_unreachable("allocShadowMem: GPUDynMemAllocKind not handled");
}

void PrepareReductionLoopGPU::reduceIntoShadowMem(Loop &loop, Value *shadow,
                                                  const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "PrepareReductionGPU: Reduce into shadow memory\n");

  // The only allowed use of the destination of the reduction that is allowed is
  // in the reduce intrinsic. This is why we can simply change the destination
  // in the intrinsic call to ensure that we reduce into shadow memory. If this
  // ever changes, we may have to do something more sophisticated here.
  redxn.call->setArgOperand(2, shadow);
}

void PrepareReductionLoopGPU::copyShadowMemToDest(Loop &loop,
                                                  GPUDynMemAllocKind dmem,
                                                  Value *shadow, Value *dest,
                                                  const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "PrepareReductionGPU: Copy shadow memory to result\n");

  LLVMContext &ctx = getContext(loop);
  Type *i64 = Type::getInt64Ty(ctx);

  Value *tt = redxn.getTTV();
  Value *size = redxn.getElemSizeV();

  BasicBlock &bb = *getExitBlockFromLatch(loop);
  IRBuilder<> builder(bb.getTerminator());

  Value *sz64 = builder.CreateIntCast(size, i64, /*isSigned=*/false);
  if (dmem == GPUDynMemAllocKind::Global)
    builder.CreateIntrinsic(Intrinsic::kit_gpu_memcpy_dtoh,
                            {tt, dest, shadow, sz64});
  else if (dmem == GPUDynMemAllocKind::UVM)
    builder.CreateMemCpyInline(dest, MaybeAlign(), shadow, MaybeAlign(), sz64);
  else
    llvm_unreachable("copyShadowMemToDest: GPUDynMemAllocKind not handled");
}

// Free the result variable on the device into which the result was computed.
void PrepareReductionLoopGPU::freeShadowMem(Loop &loop, GPUDynMemAllocKind dmem,
                                            Value *shadow,
                                            const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "PrepareReductionGPU: Free shadow memory\n");

  LLVMContext &ctx = getContext(loop);
  PointerType *mobileTy = PointerType::get(ctx, KitAS::Mobile);

  Value *tt = redxn.getTTV();

  BasicBlock &bb = *getExitBlockFromLatch(loop);
  IRBuilder<> builder(bb.getTerminator());

  if (dmem == GPUDynMemAllocKind::Global) {
    builder.CreateIntrinsic(Intrinsic::kit_gpu_free, {tt, shadow});
  } else if (dmem == GPUDynMemAllocKind::UVM) {
    Value *cst = builder.CreateAddrSpaceCast(shadow, mobileTy);
    builder.CreateIntrinsic(Intrinsic::kit_mobile_free, {tt, cst});
  } else {
    llvm_unreachable("freeShadowMem: GPUDynMemAllocKind not handled");
  }
}

bool PrepareReductionLoopGPU::run(
    Loop &loop, const SmallVectorImpl<ReductionInfo> &reductions) {
  auto sanityCheck = [](const Loop &loop) {
    assert(loop.getLoopPreheader() && "Loop must have a preheader");
    assert(getExitBlockFromLatch(loop) && "Loop must have a unique exit block");
    assert(hasTargetAttr(loop) &&
           "Loop must have the tapir.loop.target attribute");
  };

  sanityCheck(loop);
  LLVM_DEBUG(dbgs() << "PrepareReductionGPU: BEGIN '" << getName(loop)
                    << "'\n");

  GPUDynMemAllocKind dmem = getShadowMemoryKind();
  for (const ReductionInfo &redxn : reductions) {
    // We have to save this because this will change during the course of the
    // transformations.
    Value *dest = redxn.getDest();
    Value *shadow = allocShadowMem(loop, dmem, redxn);
    reduceIntoShadowMem(loop, shadow, redxn);
    copyShadowMemToDest(loop, dmem, shadow, dest, redxn);
    freeShadowMem(loop, dmem, shadow, redxn);
  }

  LLVM_DEBUG(dbgs() << "PrepareReductionGPU: END '" << getName(loop) << "'\n");

  // Mark the loop as having been prepared to ensure that we don't accidentally
  // attempt to process it more than once.
  addPreparedAttr(loop);
  return true;
}

bool llvm::detail::prepareReductionLoopForGPU(
    Loop &loop, const SmallVectorImpl<ReductionInfo> &reductions) {
  return PrepareReductionLoopGPU().run(loop, reductions);
}
