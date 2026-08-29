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
//     void f_sum(int32_t* res, int32_t v) {
//         *res += v;
//     }
//
//     void f_and(int64_t* res, int64_t v) {
//         *res &= v;
//     }
//
//     parallel_for (int i = 0; i < n; ++i) {
//         kit.reduce.0(&r_sum, sizeof(r_sum), i, 0, &f_sum);
//         kit.reduce.0(&r_and, sizeof(r_and), i, 1, &f_and);
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
//         int32_t l_sum = 0;
//         int64_t l_and = 1;
//         kit.reduce.0(&l_sum, sizeof(r_sum), i, 0, &f_sum);
//         kit.reduce.0(&l_and, sizeof(r_and), i, 1, &f_and);
//         atomicReduce(g_sum, l_sum, &sum);
//         atomicReduce(g_and, l_and, &and);
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
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/IRBuilderUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/Reductions.h"
#include "kitsune/Core/TypeUtils.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "kit-prepare"

using namespace llvm;

namespace {

// The various reduction algorithms available to be used.
enum Strategy {
  // Each thread accumulates its contribution directly into the result variable
  // using an atomic operation. This is the most naive approach possible, and is
  // unlikely to ever be useful. It may be useful when comparing approaches, so
  // it is retained.
  Direct = 1,

  // Each thread in a block writes its contribution to the result into a
  // dedicated location in shared memory. These are then reduced into a single
  // value for the block. This is accumulated into the result variable using an
  // atomic operation.
  SharedMemory,

  // Each thread in a warp computes its contribution to the result, which are
  // the reduced to a single value for the warp using warp shuffles. A single
  // thread in the warp will accumulate this directly into the result variable
  // using an atomic operation.
  WarpShuffle,

  // Each thread in a warp computes its contribution to the result, which are
  // the reduced to a single value for the warp using warp shuffles. A single
  // thread in the warp writes this to a dedicated location in shared memory.
  // These are then reduced to a single value for the block. This value is
  // accumulated into the result variable using an atomic operation.
  WarpShuffleWithSharedMemory,
};

// Transform tapir reduction loops for parallel executions on GPU's.
class PrepareReductionLoopGPU {
private:
  DominatorTree &dt;
  LoopInfo &li;
  MemorySSA &mssa;

  DomTreeUpdater dtu;
  MemorySSAUpdater mssau;

private:
  Strategy chooseStrategy(const Loop &loop,
                          const SmallVectorImpl<ReductionInfo> &redxns);

  void reduceDirect(IRBuilder<> &builder, Loop &loop, Value *globalResult,
                    Value *value, const ReductionInfo &redxn);
  void reduceSharedMemoryOnly(IRBuilder<> &builder, Loop &loop,
                              Value *globalResult, Value *value,
                              const ReductionInfo &redxn);
  void reduceWarpShuffleOnly(IRBuilder<> &builder, Loop &loop,
                             Value *globalResult, Value *value,
                             const ReductionInfo &redxn);
  void reduceWarpShuffleWithSharedMemory(IRBuilder<> &builder, Loop &loop,
                                         Value *globalResult, Value *value,
                                         const ReductionInfo &redxn);
  void reduceIntoGlobalResult(IRBuilder<> &builder, Loop &loop,
                              Value *globalResult, Value *value,
                              const ReductionInfo &redxn);

  Value *allocGlobalResult(Loop &loop, const ReductionInfo &redxn);
  Value *allocLocalResult(Loop &loop, const ReductionInfo &redxn);
  void reduceIntoLocalResult(Value *localResult, const ReductionInfo &redxn);
  void reduceIntoGlobalResult(Loop &loop, Value *globalResult,
                              Value *localResult, Strategy strategy,
                              const ReductionInfo &redxn);
  void copyGlobalResultToHost(Loop &loop, Value *globalResult,
                              const ReductionInfo &redxn);
  void freeGlobalResult(Loop &loop, Value *globalResult,
                        const ReductionInfo &redxn);

public:
  PrepareReductionLoopGPU(DominatorTree &dt, LoopInfo &li, MemorySSA &mssa)
      : dt(dt), li(li), mssa(mssa),
        dtu(dt, DomTreeUpdater::UpdateStrategy::Eager), mssau(&mssa) {}

  bool run(Loop &loop, const SmallVectorImpl<ReductionInfo> &reductions);
};

} // namespace

static cl::opt<Strategy> clStrategy(
    "tapir-gpu-reduce-mode",
    cl::desc("The strategy to use for tapir reduction loops on GPUs"),
    cl::init(Strategy::WarpShuffle), cl::value_desc("strategy"),
    cl::cat(cl::catKitClOpts),
    cl::values(clEnumValN(Strategy::Direct, "direct", ""),
               clEnumValN(Strategy::SharedMemory, "mem", ""),
               clEnumValN(Strategy::WarpShuffle, "wshf", ""),
               clEnumValN(Strategy::WarpShuffleWithSharedMemory, "wshfmem",
                          "")));

// Perform some analysis on the loop given the reductions that are performed in
// it and determine which reduction strategy to use.
Strategy PrepareReductionLoopGPU::chooseStrategy(
    const Loop &loop, const SmallVectorImpl<ReductionInfo> &redxns) {
  if (clStrategy.getNumOccurrences())
    return clStrategy;

  // TODO: Implement this.
  return Strategy::WarpShuffle;
}

// Allocate space, on the GPU, where the values being reduced will be
// accumulated. This will use Kitsune's kit.gpu.malloc intrinsic. The call will
// be added to the preheader of the reduction loop.
Value *PrepareReductionLoopGPU::allocGlobalResult(Loop &loop,
                                                  const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "PrepareReductionGPU: Allocate global result\n");

  LLVMContext &ctx = getContext(loop);
  Type *i64 = Type::getInt64Ty(ctx);

  Type *type = redxn.getType();
  Value *dest = redxn.dest;
  Value *tt = redxn.getTTV();
  Constant *one = toConstant(1L, ctx);

  BasicBlock &bb = *loop.getLoopPreheader();
  IRBuilder<> builder(bb.getTerminator());

  // Allocate memory for the global result and initialize it to the current
  // value of the final result variable. We could also initialize it to the unit
  // value, but that would require us to do an additional reduction into the
  // final result when we return from the GPU. This eliminates the need for that
  // step.
  Value *sz64 =
      builder.CreateIntCast(redxn.getElemSizeV(), i64, /*isSigned=*/false);
  Value *globalResult = createGPUMalloc(builder, redxn.tt, sz64);
  Value *globalInit = builder.CreateLoad(type, dest);
  builder.CreateIntrinsic(Intrinsic::kit_gpu_memset, {type},
                          {tt, globalResult, one, globalInit});

  return globalResult;
}

// Allocate a stack variable into which the per-thread reduction will be
// stored. The reductions may not be as simple as `res += i`, or even
// `res &= a[i]`. A single thread may contain a loop that reduces multiple times
// into the result variable `res`. For instance, something like this:
//
//     parallel_for (int i = 0; i < n; ++i) {
//       for (auto j : x)
//         for (auto k : y)
//           res += f(i, j, k);
//     }
//
// In such cases, it is better to create a local variable to accumulate all the
// values of `f` that are computed in a single thread, then accumulate the final
// result into `res`. Note that the initial value of res here is set to 0
// because that is the identity for the += operator.
//
//     parallel_for (int i = 0; i < n; ++i) {
//       decltype(res) local = 0;
//       for (auto j : x)
//         for (auto k : y)
//           local += f(i, j, k);
//       res += local;
//     }
//
// This transformation is absolutely necessary for an implementation based on
// warp shuffles since all threads in the warp must perform the shuffle. If,
// for whatever reason, we were to forgo this implementation, and reduce into
// `res` directly, this would also reduce the number of atomic reduce operations
// that would be performed on `res`.
Value *PrepareReductionLoopGPU::allocLocalResult(Loop &loop,
                                                 const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "PrepareReductionGPU: Allocate per-thread result\n");

  BasicBlock &body = *getTapirLoopDetachedBlock(loop);
  IRBuilder<> builder(&*body.begin());

  // The alloca is added to the body of the loop because, when loop-spawning
  // outlines the body, it will become a "top-level" stack variable in that
  // outlined function, which is exactly what we want.
  return detail::createLocalResult(builder, redxn, /*initialize=*/true);
}

// Change the destination of the reduce intrinsic to write to the local buffer
// \p localResult. In a later step, an atomic reduction will be performed with
// the value of this local result.
void PrepareReductionLoopGPU::reduceIntoLocalResult(
    Value *localResult, const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "PrepareReductionGPU: Reduction into local result\n");

  redxn.call->setArgOperand(2, localResult);
}

// Ensure that the values being reduced are accumulated into the result variable
// that was allocated by \ref allocResultVar. This involves replacing the calls
// to the Kitsune's reduce intrinsic with an atomic read-modify-write
// instruction if it supports the reduction operator. If it does not, a custom
// atomic reduction will be used. The original call to the reduce intrinsic
// will be removed.
void PrepareReductionLoopGPU::reduceIntoGlobalResult(
    Loop &loop, Value *globalResult, Value *localResult, Strategy strategy,
    const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "PrepareReductionGPU: Reduce into global result\n");

  Value *tt = redxn.getTTV();
  Value *op = redxn.getReduceOpV();
  Type *type = redxn.value->getType();
  Value *elemSize = redxn.getElemSizeV();
  Value *unit = redxn.unit;
  Value *reducer = redxn.reducer;

  // Once control reaches the reattach instruction, we can be certain that the
  // iteration's contribution to the final result has been computed. This will
  // have been written to the localResult variable.
  ReattachInst *reattach = getTapirLoopReattachInst(loop);
  IRBuilder<> builder(reattach);

  // Loading from the localResult is the final value that is to be accumulated
  // into the global result.
  Value *value = builder.CreateLoad(type, localResult);

  SmallVector<Type *, 2> overloadTys = redxn.getOverloadTypes();
  SmallVector<Value *, 8> args = {tt,    op,   globalResult, elemSize,
                                  value, unit, reducer};
  args.append(redxn.getExtraArgs());

  auto getIntrinsic = [](Strategy strategy) -> Intrinsic::ID {
    switch (strategy) {
    case Strategy::Direct: return Intrinsic::kit_gpu_reduce_direct;
    case Strategy::SharedMemory: return Intrinsic::kit_gpu_reduce_shared_memory;
    case Strategy::WarpShuffle: return Intrinsic::kit_gpu_reduce_warp_shuffle;
    case Strategy::WarpShuffleWithSharedMemory:
      return Intrinsic::kit_gpu_reduce_warp_shuffle_shared_memory;
    }
    llvm_unreachable("reduceIntoGlobalResult: Strategy not handled");
  };
  builder.CreateIntrinsic(getIntrinsic(strategy), overloadTys, args);
}

// Once the reduction has been computed, copy the result from the device to the
// final destination on the host.
void PrepareReductionLoopGPU::copyGlobalResultToHost(
    Loop &loop, Value *globalResult, const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "PrepareReductionGPU: Copy global result to host\n");

  LLVMContext &ctx = getContext(loop);
  Type *i64 = Type::getInt64Ty(ctx);

  Value *tt = redxn.getTTV();
  Value *dest = redxn.dest;
  Value *elemSize = redxn.getElemSizeV();

  BasicBlock &bb = *getExitBlockFromLatch(loop);
  IRBuilder<> builder(bb.getTerminator());

  Value *sz64 = builder.CreateIntCast(elemSize, i64, /*isSigned=*/true);
  builder.CreateIntrinsic(Intrinsic::kit_gpu_memcpy_dtoh,
                          {tt, dest, globalResult, sz64});
}

// Free the result variable on the device into which the result was computed.
void PrepareReductionLoopGPU::freeGlobalResult(Loop &loop, Value *globalResult,
                                               const ReductionInfo &redxn) {
  LLVM_DEBUG(dbgs() << "PrepareReductionGPU: Free partial buffer\n");

  Value *tt = redxn.getTTV();

  BasicBlock &bb = *getExitBlockFromLatch(loop);
  IRBuilder<> builder(bb.getTerminator());
  builder.CreateIntrinsic(Intrinsic::kit_gpu_free, {tt, globalResult});
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

  Strategy strategy = chooseStrategy(loop, reductions);
  for (const ReductionInfo &redxn : reductions) {
    Value *globalResult = allocGlobalResult(loop, redxn);
    Value *localResult = allocLocalResult(loop, redxn);
    reduceIntoLocalResult(localResult, redxn);
    reduceIntoGlobalResult(loop, globalResult, localResult, strategy, redxn);
    copyGlobalResultToHost(loop, globalResult, redxn);
    freeGlobalResult(loop, globalResult, redxn);
  }

  LLVM_DEBUG(dbgs() << "PrepareReductionGPU: END '" << getName(loop) << "'\n");

  // Mark the loop as having been prepared to ensure that we don't accidentally
  // attempt to process it more than once.
  addPreparedAttr(loop);
  return true;
}

bool llvm::detail::prepareReductionLoopForGPU(
    Loop &loop, const SmallVectorImpl<ReductionInfo> &reductions,
    DominatorTree &dt, LoopInfo &li, MemorySSA &mssa) {
  return PrepareReductionLoopGPU(dt, li, mssa).run(loop, reductions);
}
