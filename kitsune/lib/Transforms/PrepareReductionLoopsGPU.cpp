//===- PrepareReductionLoopsGPU.cpp - Transform reduction loops for GPU ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transform tapir loops that perform reductions to a form that is suitable for
// parallel execution.
//
// Consider the loop shown below
//
//     int32_t r_sum = 0;
//     int64_t r_and = 1;
//     parallel_for (int i = 0; i < n; ++i) {
//         r_sum += i;
//         r_and &= 1;
//     }
//
// Frontends are expected to use the kit.reduce.0 intrinsic to represent the
// loop above
//
//     void sum(int32_t* res, int32_t v) {
//         *res += v;
//     }
//
//     void and(int64_t* res, int64_t v) {
//         *res &= v;
//     }
//
//     parallel_for (int i = 0; i < n; ++i) {
//         kit.reduce.0(&r_sum, sizeof(r_sum), i, 0, &sum);
//         kit.reduce.0(&r_and, sizeof(r_and), i, 1, &and);
//     }
//
// This pass will transform this into the following for parallel execution on a
// GPU.
//
//     int32_t *g_sum = (int32_t*)kit.gpu.malloc(sizeof(int32_t));
//     *g_sum = 0;
//     int64_t *g_and = (int64_t*)kit.gpu.malloc(sizeof(int64_t));
//     *g_and = 1;
//     parallel_for (int i = 0; j < n; ++i) {
//         atomicReduce(&sum, g_sum, i);
//         atomicReduce(&and, g_and, i);
//     }
//     kit.gpu.memcpy.dtoh(&r_sum, g_sum, sizeof(int32_t));
//     kit.gpu.memcpy.dtoh(&r_and, g_and, sizeof(int64_t));
//     kit.gpu.free(g_sum);
//     kit.gpu.free(g_and);
//
// Here, some memory is allocated on the GPU, and the entire reduction is
// directly written to that location using atomics.

// FIXME: Obviously, the performance of this will be absolutely dreadful, but
// as a first implementation using atomics, it will at least be correct.
//
//===----------------------------------------------------------------------===//

#include "PrepareReductionLoops.h"
#include "kitsune/Core/AddrSpace.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/IRBuilderUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/Reductions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "kit-prepare"

using namespace llvm;

namespace {

// Transform tapir reduction loops for parallel executions on GPU's.
class PrepareReductionLoopGPU {
private:
  Value *allocResultVar(Loop &loop, const ReductionInfo &redxn);
  void reduceIntoResultVar(Loop &loop, Value *resultVar,
                           const ReductionInfo &redxn);
  void copyResultVarToHost(Loop &loop, Value *resultVar,
                           const ReductionInfo &redxn);
  void freeResultVar(Loop &loop, Value *resultVar, const ReductionInfo &redxn);
  void eraseReduceCall(Loop &loop, const ReductionInfo &redxn);

public:
  PrepareReductionLoopGPU() = default;

  bool run(Loop &loop, const SmallVectorImpl<ReductionInfo> &reductions);
};

} // namespace

// Allocate space, on the GPU, where the values being reduced will be
// accumulated. This will use Kitsune's kit.gpu.malloc intrinsic. The call will
// be added to the preheader of the reduction loop.
Value *PrepareReductionLoopGPU::allocResultVar(Loop &loop,
                                               const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "PrepareReductionGPU: Allocate buffer for partial\n");

  LLVMContext &ctx = getContext(loop);
  Type *i64 = Type::getInt64Ty(ctx);

  BasicBlock &bb = *loop.getLoopPreheader();
  IRBuilder<> builder(bb.getTerminator());
  Value *sz64 =
      builder.CreateIntCast(redxn.getElemSizeV(), i64, /*isSigned=*/false);
  Value *result = createGPUMalloc(builder, redxn.tt, sz64);
  builder.CreateStore(redxn.unit, result);

  return result;
}

// Ensure that the values being reduced are accumulated into the result variable
// that was allocated by \ref allocResultVar. This involves replacing the calls
// to the Kitsune's reduce intrinsic with an atomic read-modify-write
// instruction if it supports the reduction operator. If it does not, a custom
// atomic reduction will be used. The original call to the reduce intrinsic
// will be removed.
void PrepareReductionLoopGPU::reduceIntoResultVar(Loop &loop, Value *resultVar,
                                                  const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "PrepareReductionGPU: Reduce into partial\n");

  std::optional<AtomicRMWInst::BinOp> atomicOp = getAtomicOp(redxn.reduceOp);
  if (!atomicOp)
    emitDiagnostic(loop, DiagID::ErrNYI,
                   "Reduction operator not supported by AtomicRMWInst");

  Module &m = *getModule(loop);
  const DataLayout &dl = m.getDataLayout();
  Align align = dl.getPointerABIAlignment(KitAS::Default);

  IRBuilder<> builder(redxn.call);
  builder.CreateAtomicRMW(*atomicOp, resultVar, redxn.value, align,
                          AtomicOrdering::Monotonic);
}

// Once the reduction has been computed, copy the result from the device to the
// final destination on the host.
void PrepareReductionLoopGPU::copyResultVarToHost(Loop &loop, Value *resultVar,
                                                  const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "PrepareReductionGPU: Copy result to host\n");

  LLVMContext &ctx = getContext(loop);
  Type *i64 = Type::getInt64Ty(ctx);

  BasicBlock &bb = *getExitBlockFromLatch(loop);
  IRBuilder<> builder(bb.getTerminator());
  Value *sz64 =
      builder.CreateIntCast(redxn.getElemSizeV(), i64, /*isSigned=*/true);
  builder.CreateIntrinsic(Intrinsic::kit_gpu_memcpy_dtoh,
                          {redxn.getTTV(), redxn.dest, resultVar, sz64});
}

// Free the result variable on the device into which the result was computed.
void PrepareReductionLoopGPU::freeResultVar(Loop &loop, Value *resultVar,
                                            const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "PrepareReductionGPU: Free partial buffer\n");

  BasicBlock &bb = *getExitBlockFromLatch(loop);
  IRBuilder<> builder(bb.getTerminator());
  builder.CreateIntrinsic(Intrinsic::kit_gpu_free, {redxn.getTTV(), resultVar});
}

// By this point, we know that the reduction is being computed with an
// atomicrmw instruction (or the equivalent). The reduce call can, therefore,
// be removed.
void PrepareReductionLoopGPU::eraseReduceCall(Loop &loop,
                                              const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "PrepareReductionGPU: Erase original reduce call\n");

  redxn.call->eraseFromParent();
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

  for (const ReductionInfo &redxn : reductions) {
    Value *result = allocResultVar(loop, redxn);
    reduceIntoResultVar(loop, result, redxn);
    copyResultVarToHost(loop, result, redxn);
    freeResultVar(loop, result, redxn);
    eraseReduceCall(loop, redxn);
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
