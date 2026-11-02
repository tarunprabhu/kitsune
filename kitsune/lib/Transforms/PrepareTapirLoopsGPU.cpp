//===- PrepareTapirLoopsGPU.cpp - Transform tapir loops for GPU -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transform a tapir loop nest for execution on a GPU.
//
// Nests that do not perform reductions are not transformed.
//
// For nests that perform reduction, shadow memory is allocated on the GPU into
// which the GPU reduction will be accumulated. The reduction intrinsics are
// modified to write to this location. The "shadow" memory is allocated in UVM
// by default, but may also be allocated directly in global memory.
//
//     int64_t r_sum = 0;
//     int32_t r_and = 0;
//     parallel_for (int i = 0; i < n; ++i) {
//         kit.reduce(&r_sum, sizeof(r_sum), i, 0, &f_sum);
//         kit.reduce(&r_and, sizeof(r_and), i, 1, &f_and);
//     }
//
// This pass will transform this into the following for parallel execution on a
// GPU.
//
//     int64_t numThreads = kit.num.cpu.threads();
//     int64_t itersPerThread = (n + numThreads - 1) / numThreads;
//     int32_t *g_sum = allocateShadowMemory<int32_t>(0);
//     int64_t *g_and = allocateShadowMemory<int64_t>(0);
//     parallel_for (int j = 0; j < numThreads; ++j) {
//         kit.reduce(&g_sum, sizeof(r_sum), i, 0, &f_sum);
//         kit.reduce(&g_and, sizeof(r_and), i, 1, &f_and);
//     }
//     copyToFinalResult(&r_sum, g_sum, sizeof(int32_t));
//     copyToFinalResult(&r_and, g_and, sizeof(int64_t));
//     freeShadowMemory(g_sum);
//     freeShadowMemory(g_and);
//
// This is the first in a multi-stage process of transforming such a nest.
// Passes that run later in the pipeline will perform more transformations.
//
//===----------------------------------------------------------------------===//

#include "PrepareTapirLoopsImpl.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/GPUMemUtils.h"
#include "kitsune/Core/IRBuilderUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/Reductions.h"
#include "kitsune/Core/TypeUtils.h"
#include "kitsune/Support/CommandLineOptions.h"

#define DEBUG_TYPE "kit-prepare"

using namespace llvm;

static cl::opt<GPUDynMemAllocKind>
    clShadow("tapir-gpu-reduce-shadow",
             cl::desc("Where to allocate the shadow reduction destination"),
             cl::init(GPUDynMemAllocKind::UVM), cl::value_desc("where"),
             cl::cat(cl::catKitClOpts),
             cl::values(clEnumValN(GPUDynMemAllocKind::Global, "global", ""),
                        clEnumValN(GPUDynMemAllocKind::UVM, "uvm", "")));

namespace {

// Base class to work with shadow memory.
class IShadowMem {
protected:
  LLVMContext &ctx;
  Type *i64 = nullptr;
  PointerType *ptr = nullptr;
  PointerType *mobileTy = nullptr;

protected:
  IShadowMem(LLVMContext &ctx);

public:
  virtual Value *alloc(IRBuilder<> &builder, const ReductionInfo &redxn) = 0;
  virtual void copy(IRBuilder<> &builder, Value *shadow, Value *dest,
                    const ReductionInfo &redxn) = 0;
  virtual void free(IRBuilder<> &builder, Value *shadow,
                    const ReductionInfo &redxn) = 0;
};

// Implementation class when the shadow memory uses GPU global memory.
class ShadowMemGlobal : public IShadowMem {
public:
  ShadowMemGlobal(LLVMContext &ctx);
  virtual ~ShadowMemGlobal() = default;

  virtual Value *alloc(IRBuilder<> &builder,
                       const ReductionInfo &redxn) override final;
  virtual void copy(IRBuilder<> &builder, Value *shadow, Value *dest,
                    const ReductionInfo &redxn) override final;
  virtual void free(IRBuilder<> &builder, Value *shadow,
                    const ReductionInfo &redxn) override final;
};

// Implementation class when the shadow memory uses UVM.
class ShadowMemUVM : public IShadowMem {
public:
  ShadowMemUVM(LLVMContext &ctx);
  virtual ~ShadowMemUVM() = default;

  virtual Value *alloc(IRBuilder<> &builder,
                       const ReductionInfo &redxn) override final;
  virtual void copy(IRBuilder<> &builder, Value *shadow, Value *dest,
                    const ReductionInfo &redxn) override final;
  virtual void free(IRBuilder<> &builder, Value *shadow,
                    const ReductionInfo &redxn) override final;
};

} // namespace

//==------------------------------ IShadowMem ------------------------------==//

IShadowMem::IShadowMem(LLVMContext &ctx)
    : ctx(ctx), i64(Type::getInt64Ty(ctx)), ptr(PointerType::getUnqual(ctx)),
      mobileTy(getMobileTy(ctx)) {}

//==--------------------------- ShadowMemGlobal ----------------------------==//

ShadowMemGlobal::ShadowMemGlobal(LLVMContext &ctx) : IShadowMem(ctx) {}

Value *ShadowMemGlobal::alloc(IRBuilder<> &builder,
                              const ReductionInfo &redxn) {
  Value *tt = redxn.getTTV();
  Value *dest = redxn.getDest();
  Value *size = redxn.getElemSizeV();

  Value *sz64 = builder.CreateIntCast(size, i64, /*isSigned=*/false);
  Value *shadow = createGPUMalloc(builder, redxn.tt, sz64);
  builder.CreateIntrinsic(Intrinsic::kit_gpu_memcpy_htod,
                          {tt, shadow, dest, sz64});
  return shadow;
}

void ShadowMemGlobal::copy(IRBuilder<> &builder, Value *shadow, Value *dest,
                           const ReductionInfo &redxn) {
  Value *size = redxn.getElemSizeV();
  Value *tt = redxn.getTTV();

  Value *sz64 = builder.CreateIntCast(size, i64, /*isSigned=*/false);
  builder.CreateIntrinsic(Intrinsic::kit_gpu_memcpy_dtoh,
                          {tt, dest, shadow, sz64});
}

void ShadowMemGlobal::free(IRBuilder<> &builder, Value *shadow,
                           const ReductionInfo &redxn) {
  Value *tt = redxn.getTTV();
  builder.CreateIntrinsic(Intrinsic::kit_gpu_free, {tt, shadow});
}

//==----------------------------- ShadowMemUVM -----------------------------==//

ShadowMemUVM::ShadowMemUVM(LLVMContext &ctx) : IShadowMem(ctx) {}

Value *ShadowMemUVM::alloc(IRBuilder<> &builder, const ReductionInfo &redxn) {
  Value *tt = redxn.getTTV();
  Value *dest = redxn.getDest();
  Value *size = redxn.getElemSizeV();

  Value *sz64 = builder.CreateIntCast(size, i64, /*isSigned=*/false);
  Value *b = builder.CreateIntrinsic(Intrinsic::kit_mobile_alloc, {tt, sz64});
  Value *shadow = builder.CreateAddrSpaceCast(b, ptr);

  builder.CreateMemCpyInline(shadow, MaybeAlign(), dest, MaybeAlign(), sz64);

  return shadow;
}

void ShadowMemUVM::copy(IRBuilder<> &builder, Value *shadow, Value *dest,
                        const ReductionInfo &redxn) {
  Value *size = redxn.getElemSizeV();

  Value *sz64 = builder.CreateIntCast(size, i64, /*isSigned=*/false);
  builder.CreateMemCpyInline(dest, MaybeAlign(), shadow, MaybeAlign(), sz64);
}

void ShadowMemUVM::free(IRBuilder<> &builder, Value *shadow,
                        const ReductionInfo &redxn) {
  Value *tt = redxn.getTTV();
  Value *cst = builder.CreateAddrSpaceCast(shadow, mobileTy);
  builder.CreateIntrinsic(Intrinsic::kit_mobile_free, {tt, cst});
}

//==------------------------------------------------------------------------==//

// Allocate space, on the GPU, where the values being reduced will be
// accumulated. The allocated memory will be initialized with the value in the
// destination of the reduction.
static Value *allocShadowMem(IShadowMem *iShadowMem, Loop &root,
                             const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "PrepareTapirReductionGPU: Allocate shadow memory\n");

  BasicBlock &bb = *root.getLoopPreheader();
  IRBuilder<> builder(bb.getTerminator());

  return iShadowMem->alloc(builder, redxn);
}

static void reduceIntoShadowMem(Value *shadow, const ReductionInfo &redxn) {
  // The only allowed use of the destination of the reduction that is allowed
  // is in the reduce intrinsic. This is why we can simply change the
  // destination in the intrinsic call to ensure that we reduce into shadow
  // memory. If this ever changes, we may have to do something more
  // sophisticated here.
  redxn.call->setArgOperand(ReductionInfo::argNumDest, shadow);
}

static void copyShadowMemToDest(IShadowMem *iShadowMem, Loop &root,
                                Value *shadow, Value *dest,
                                const ReductionInfo &redxn) {
  LLVM_DEBUG(
      dbgs() << "PrepareTapirReductionGPU: Copy shadow memory to result\n");

  BasicBlock &bb = *getUniqueNonDeadEndExitBlock(root);
  IRBuilder<> builder(bb.getTerminator());

  iShadowMem->copy(builder, shadow, dest, redxn);
}

// Free the result variable on the device into which the result was computed.
static void freeShadowMem(IShadowMem *iShadowMem, Loop &root, Value *shadow,
                          const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "PrepareTapirReductionGPU: Free shadow memory\n");

  BasicBlock &bb = *getUniqueNonDeadEndExitBlock(root);
  IRBuilder<> builder(bb.getTerminator());

  iShadowMem->free(builder, shadow, redxn);
}

// Get the shadow memory kind to use. We always use UVM unless an override has
// been provided explicitly. Using UVM may result in page migrations between
// device and host, so it may not always be profitable. However, a profitability
// analysis requires at least a function analysis, so we do not do that here.
static GPUDynMemAllocKind
getShadowMemoryKind([[maybe_unused]] const Loop &root) {
  if (clShadow.getNumOccurrences()) {
    if (clShadow == GPUDynMemAllocKind::Global)
      // The copy from GPU global memory to the final result is added to the
      // loop exit block. This might end up happening before the kernel is
      // synchronized, which might result in the wrong value being copied. At
      // some point, we will settle on exactly how to handle asynchrony in
      // kernel launches, at which point, it will be safe to use this.
      emitDiagnostic(DiagID::WarnGeneric,
                     "Using global memory for the shadow reduction variable is "
                     "known to be buggy and NOT recommended");
    return clShadow;
  }
  return GPUDynMemAllocKind::UVM;
}

static std::unique_ptr<IShadowMem> getShadowMemImpl(const Loop &root) {
  LLVMContext &ctx = getContext(root);

  switch (getShadowMemoryKind(root)) {
  case GPUDynMemAllocKind::Global:
    return std::make_unique<ShadowMemGlobal>(ctx);
  case GPUDynMemAllocKind::UVM: //
    return std::make_unique<ShadowMemUVM>(ctx);
  default: break;
  }
  llvm_unreachable("getShadowMemImpl: GPUDynMemAllocKind not handled");
}

static void markNestPrepared(const SmallVectorImpl<Loop *> &nest) {
  for (Loop *loop : nest)
    addPreparedAttr(*loop);
}

static bool prepareReductionNest(const SmallVectorImpl<Loop *> &nest) {
  LLVM_DEBUG(dbgs() << "PrepareTapirReductionGPU: BEGIN '" << getName(*nest[0])
                    << "'\n");

  Loop &root = *nest[0];
  SmallVector<ReductionInfo, 1> redxns = collectReductions(root);
  std::unique_ptr<IShadowMem> iShadowMem = getShadowMemImpl(root);
  for (const ReductionInfo &redxn : redxns) {
    // We have to save this because this will change during the course of the
    // transformations.
    Value *dest = redxn.getDest();
    Value *shadow = allocShadowMem(&*iShadowMem, root, redxn);
    reduceIntoShadowMem(shadow, redxn);
    copyShadowMemToDest(&*iShadowMem, root, shadow, dest, redxn);
    freeShadowMem(&*iShadowMem, root, shadow, redxn);
  }

  markNestPrepared(nest);

  LLVM_DEBUG(dbgs() << "PrepareTapirReductionGPU: END '" << getName(root)
                    << "'\n");
  return true;
}

static bool prepareParallelNest(const SmallVectorImpl<Loop *> &nest) {
  const Loop &root = *nest[0];
  LLVM_DEBUG(dbgs() << "PrepareTapirLoopGPU: BEGIN '" << getName(root)
                    << "'\n");

  // Nothing needs to be done to prepare a parallel loop nest for execution on a
  // GPU. Just mark all loops in the nest as prepared.
  markNestPrepared(nest);

  LLVM_DEBUG(dbgs() << "PrepareTapirLoopGPU: END '" << getName(root) << "'\n");

  return true;
}

bool llvm::detail::prepareTapirNestGPU(Loop &root) {
  SmallVector<Loop *, 4> loops = getTapirLoopsInNest(root);
  if (hasReductionAttr(root))
    return prepareReductionNest(loops);
  else
    return prepareParallelNest(loops);
}
