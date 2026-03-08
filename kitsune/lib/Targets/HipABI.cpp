//===- HipABI.cpp - Tapir target for Kitsune's hip runtime ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// Copyright (c) 2021, 2023, 2025 Los Alamos National Security, LLC.
//  All rights reserved.
//
// Copyright 2021, 2023, 2025. Los Alamos National Security, LLC. This
//  software was produced under U.S. Government contract
//  DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which
//  is operated by Los Alamos National Security, LLC for the
//  U.S. Department of Energy. The U.S. Government has rights to use,
//  reproduce, and distribute this software.  NEITHER THE GOVERNMENT
//  NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS
//  OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.
//  If software is modified to produce derivative works, such modified
//  software should be clearly marked, so as not to confuse it with
//  the version available from LANL.
//
//  Additionally, redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the
//  following conditions are met:
//
// Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of conditions and the following
//      disclaimer in the documentation and/or other materials provided
//      with the distribution.
//
//    * Neither the name of Los Alamos National Security, LLC, Los
//      Alamos National Laboratory, LANL, the U.S. Government, nor the
//      names of its contributors may be used to endorse or promote
//      products derived from this software without specific prior
//      written permission.
//
//  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
//  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
//  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
//
// Tapir target that lowers to Kitsune's cuda runtime
//
//===----------------------------------------------------------------------===//

#include "kitsune/Targets/HipABI.h"
#include "GPUTTUtils.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/KernelProperties.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/TTOptions.h"
#include "kitsune/Core/TargetUtils.h"
#include "kitsune/Frontend/CommandLineOptions.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "hipabi"

// For some background material see the AMDGPU target documentation
// at: https://llvm.org/docs/AMDGPUUsage.html
//
// This transformation is carrying out the prep to convert Tapir to a kernel
// module suitable for codegen using the AMDGPU target.

static cl::opt<unsigned> defaultGrainsize(
    "hipabi-default-grainsize", cl::init(1), cl::Hidden,
    cl::desc("The default grain size used by the transform "
             "when analysis fails to determine one. (default=1)"),
    cl::cat(cl::catKitClDevOpts));

// FIXME: We really should not be exposing command line options from other
// source files. This is an experimental option that has been hacked in for the
// moment. If this is useful, we should consider adding it to the tapir target
// options instead. Otherwise, it should be removed altogether.
cl::opt<bool> clUseYLaunch("hipabi-y-launch", cl::init(false), cl::Hidden,
                           cl::desc("Launch kernel using y-axis threading"),
                           cl::cat(cl::catKitClDevOpts));

static constexpr StringRef HIPABI_PREFIX = "__kithip_";
static constexpr StringRef HIPABI_KERNEL_NAME_PREFIX = "__kithip_loop_";

/// The loop outline process for transforming a Tapir parallel loop into a
/// hip kernel function.
/// \ingroup kitsune
class HipLoop : public LoopOutlineProcessor {
private:
  /// The name of the kernel into which the loop is outlined.
  std::string kernelName;

  /// For GPU targets, we outline the loop into a separate module. This is that
  /// module.
  Module &kernelModule;

  // AMDGCN intrinsics.
  FunctionCallee hipWorkItemIdFn;
  FunctionCallee hipWorkItemIdXFn, hipWorkItemIdYFn, hipWorkItemIdZFn;
  FunctionCallee hipWorkGroupIdFn;
  FunctionCallee hipWorkGroupIdXFn, hipWorkGroupIdYFn, hipWorkGroupIdZFn;
  FunctionCallee hipBlockDimFn;

  /// The GlobalValue's used in the loop that is being outlined. This includes
  /// functions, global variables, aliases and ifunc's.
  SmallSet<GlobalValue *, 8> usedGlobalValues;

private:
  Value *emitWorkItemId(IRBuilder<> &builder, int itemIndex);
  Value *emitWorkGroupId(IRBuilder<> &builder, int itemIndex);
  Value *emitWorkGroupSize(IRBuilder<> &builder, int itemIndex);

public:
  /// @brief Build the HipLoop outline processor.
  /// @param M: Module containing the input code.
  /// @param KM: The module that will contain the generated kernel.
  /// @param KernelName: The name of the kernel function that is generated.
  /// @param TTO: The tapir target options.
  HipLoop(Module &hostM, Module &kernelModule, StringRef kernelName,
          const TTOptions &tto);
  ~HipLoop();

  /// Process the TapirLoop before it is outlined -- just prior to the
  /// outlining occurs.  This allows the VMap and related details to be
  /// customized prior to outlining related operations (e.g. cloning of
  /// LLVM constructs).
  void preProcessTapirLoop(TapirLoopInfo &tl, ValueToValueMapTy &vmap) override;

  /// Processes an outlined Function Helper for a Tapir loop, just after the
  /// function has been outlined.
  void postProcessOutline(TapirLoopInfo &tl, TaskOutlineInfo &toi,
                          ValueToValueMapTy &vmap) override;

  /// Processes a call to an outlined Function Helper for a Tapir loop.
  void processOutlinedLoopCall(TapirLoopInfo &tl, TaskOutlineInfo &toi,
                               DominatorTree &dt) override;
};

/// @brief Return the work item ID for the calling thread. (thread index)
/// @param Builder - IR builder for code gen assistance.
/// @param ItemIndex - which work item dimension (x=0,y=1,z=2)
/// @param Low - Low-end of value range if known.
/// @param High -- High-end of value range if known.
Value *HipLoop::emitWorkItemId(IRBuilder<> &builder, int itemIndex) {
  switch (itemIndex) {
  case 0:
    return builder.CreateCall(hipWorkItemIdXFn, {}, ".kern.witem.x");
  case 1:
    return builder.CreateCall(hipWorkItemIdYFn, {}, ".kern.witem.y");
  case 2:
    return builder.CreateCall(hipWorkItemIdZFn, {}, ".kern.witem.z");
  default:
    llvm_unreachable("unexpected item index!");
  }
}

/// @brief Return the work group ID for the calling thread. (block index)
/// @param Builder - IR builder for code gen assistance.
/// @param ItemIndex - which work item dimension (x=0,y=1,z=2)
Value *HipLoop::emitWorkGroupId(IRBuilder<> &builder, int itemIndex) {
  switch (itemIndex) {
  case 0:
    return builder.CreateCall(hipWorkGroupIdXFn, {}, ".kern.wgroup.x");
  case 1:
    return builder.CreateCall(hipWorkGroupIdYFn, {}, ".kern.wgroup.y");
  case 2:
    return builder.CreateCall(hipWorkGroupIdZFn, {}, ".kern.wgroup.z");
  default:
    llvm_unreachable("unexpected item index!");
  }
}

/// @brief Return the work group size for the calling thread. (block size)
/// @param Builder - IR builder for code gen assistance.
/// @param ItemIndex - which work item dimension (x=0,y=1,z=2)
Value *HipLoop::emitWorkGroupSize(IRBuilder<> &builder, int itemIndex) {
  auto getName = [](int itemIndex) -> StringRef {
    switch (itemIndex) {
    case 0:
      return ".kern.blkdim.x";
    case 1:
      return ".kern.blkdim.y";
    case 2:
      return ".kern.blkdim.z";
    default:
      llvm_unreachable("emitWorkGroupSize: unexpected item index!");
    };
  };

  LLVMContext &ctx = builder.getContext();
  Constant *index = ConstantInt::get(Type::getInt32Ty(ctx), itemIndex);
  StringRef name = getName(itemIndex);

  return builder.CreateCall(hipBlockDimFn, {index}, name);
}

HipLoop::HipLoop(Module &hostM, Module &kernelModule, StringRef kernelName,
                 const TTOptions &tto)
    : LoopOutlineProcessor(hostM, kernelModule, tto,
                           CloneFunctionChangeType::DifferentModule),
      kernelName(kernelName), kernelModule(kernelModule) {
  LLVM_DEBUG(dbgs() << "hipabi: hip loop outliner creation:\n"
                    << "\ttransforming loop to kernel: " << kernelName
                    << "(...)\n"
                    << "\tdevice-side module name    : "
                    << kernelModule.getName() << "\n\n");

  LLVMContext &ctx = kernelModule.getContext();
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);

  // We use ROCm/HSA/HIP entry points for various runtime calls.  These calls
  // are often at a lower level vs. user-facing entry points.  This follows
  // lower-level code generation details for HIP (that also include details
  // tucked into the HIP-centric header files as well a Clang lowering).

  // Get the local workitem ID for the calling thread.
  hipWorkItemIdFn = kernelModule.getOrInsertFunction(
      "__ockl_get_local_id",
      i64,  // return local thread id.
      i32); // axis/index select (x=0, y=1, z=2).

  // Get the work group ID for the calling thread.
  hipWorkGroupIdFn = kernelModule.getOrInsertFunction(
      "__ockl_get_group_id",
      i64,  // return local thread id.
      i32); // axis/index select (x=0, y=1, z=2).

  // Get the block size for the calling thread.
  hipBlockDimFn = kernelModule.getOrInsertFunction(
      "__ockl_get_local_size",
      i64,  // return local thread id.
      i32); // axis/index select (x=0, y=1, z=2).

  /* threadIdx.x */
  hipWorkItemIdXFn = Intrinsic::getOrInsertDeclaration(
      &kernelModule, Intrinsic::amdgcn_workitem_id_x);
  /* threadIdx.y */
  hipWorkItemIdYFn = Intrinsic::getOrInsertDeclaration(
      &kernelModule, Intrinsic::amdgcn_workitem_id_y);
  /* threadIdx. z */
  hipWorkItemIdZFn = Intrinsic::getOrInsertDeclaration(
      &kernelModule, Intrinsic::amdgcn_workitem_id_z);

  /* blockIdx.x */
  hipWorkGroupIdXFn = Intrinsic::getOrInsertDeclaration(
      &kernelModule, Intrinsic::amdgcn_workgroup_id_x);
  /* blockIdx.y */
  hipWorkGroupIdYFn = Intrinsic::getOrInsertDeclaration(
      &kernelModule, Intrinsic::amdgcn_workgroup_id_y);
  /* blockIdx.z */
  hipWorkGroupIdZFn = Intrinsic::getOrInsertDeclaration(
      &kernelModule, Intrinsic::amdgcn_workgroup_id_z);
}

HipLoop::~HipLoop() { /* no-op */ }

void HipLoop::preProcessTapirLoop(TapirLoopInfo &tl, ValueToValueMapTy &vmap) {
  bool verboseMode = getOptions().getTapirVerbose();
  if (verboseMode) {
    errs() << "kitsune[hipabi]: pre-processing tapir loop.\n";
    errs() << "  - collecting global values from loop...\n";
  }

  // Collect the top-level entities (Function, GlobalVariable, GlobalAlias
  // and GlobalIFunc) that are used in the outlined loop. Since the outlined
  // loop will live in the kernelModule, any GlobalValue's used in it must be
  // be cloned into the kernelModule and then registered with the cuda runtime.
  // The registration will be done in the global ctor which will be generated by
  // a later pass.
  collectGlobalValues(*tl.getLoop(), usedGlobalValues);

  // HIP appears to require protected visibility. Without this, attempting to
  // link the fat binary results in a relocation error.
  cloneUsedGlobalVariablesInto(
      kernelModule, usedGlobalValues, vmap,
      /* address space for constant globals */ AMDGPUAS::CONSTANT_ADDRESS,
      /* address space for non-const globals */ AMDGPUAS::GLOBAL_ADDRESS,
      /* visibility for constant globals */ GlobalValue::DefaultVisibility,
      /* visibility for non-const globals */ GlobalValue::ProtectedVisibility);

  // The global variables have to be cloned before cloning the functions because
  // they may be used in the bodies of functions to be cloned.
  cloneReachableFuncsInto(kernelModule, usedGlobalValues, vmap);
  cloneReachableIFuncsInto(kernelModule, usedGlobalValues, vmap);

  // The aliasee in global aliases is a global value, so they must be cloned
  // after the global variables and functions are in the vmap.
  cloneUsedGlobalAliasesInto(kernelModule, usedGlobalValues, vmap);
}

void HipLoop::postProcessOutline(TapirLoopInfo &tl, TaskOutlineInfo &Out,
                                 ValueToValueMapTy &vmap) {
  Task *task = tl.getTask();
  Loop *loop = tl.getLoop();

  BasicBlock *bbEntry = cast<BasicBlock>(vmap[loop->getLoopPreheader()]);
  BasicBlock *bbHeader = cast<BasicBlock>(vmap[loop->getHeader()]);
  BasicBlock *bbExit = cast<BasicBlock>(vmap[tl.getExitBlock()]);
  PHINode *iv = cast<PHINode>(vmap[tl.getPrimaryInduction().first]);
  Type *ivType = iv->getType();

  // We no longer need the cloned sync region.
  auto *clonedSyncReg =
      cast<Instruction>(vmap[task->getDetach()->getSyncRegion()]);
  clonedSyncReg->eraseFromParent();

  // Get the kernel function for this loop and clean up any stray (target
  // related) attributes. Because of the way we compile the code, those
  // attributes will only be relevant for the host.
  Function *kernelF = Out.Outline;
  kernelF->setName(kernelName);

  // Remove any attributes that are only relevant for the host.
  kernelF->removeFnAttr("target-cpu");
  kernelF->removeFnAttr("target-features");
  kernelF->removeFnAttr("tune-cpu");

  // Remove other attributes that we cannot deal with in any reasonable way in
  // the device
  kernelF->removeFnAttr(Attribute::UWTable);

  // Add an attribute identifying this as a function outlined from a tapir loop.
  kernelF->addFnAttr(Attribute::KitKernel);

  // Add new target-specific attributes
  kernelF->addFnAttr("target-cpu", getOptions().getHipArch());
  kernelF->addFnAttr("target-features", getOptions().getHipTargetFeatures());

  // Add other attributes that are either required or desirable.
  kernelF->addFnAttr("no-trapping-math", "true");
  kernelF->addFnAttr(Attribute::MustProgress);
  kernelF->addFnAttr(Attribute::NoUnwind);

  // This only works when the code object version >= 5, but we have ensured that
  // this is the case in the frontend.
  kernelF->addFnAttr("uniform-work-group-size", "true");

  // Specify the minimum and maximum flat work group sizes that will be used
  // when the kernel is dispatched.
  std::string attrVal = "128,1024";
  kernelF->addFnAttr("amdgpu-flat-work-group-size", attrVal);

  // AMD requires that the kernel function have protected visiblity otherwise
  // AMD's runtime is unable to find the kernel function at runtime. This, in
  // turn requires the function to have external linkage. In case the function
  // gets here with a different linkage type, just override it.
  kernelF->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
  kernelF->setVisibility(GlobalValue::VisibilityTypes::ProtectedVisibility);
  kernelF->setCallingConv(CallingConv::AMDGPU_KERNEL);

#if 0 // DISABLED FOR TESTING
  unsigned maxThreadsPerBlock = getOptions().getMaxThreadsPerBlock();
  unsigned defaultThreadsPerBlock = getOptions().getFixedThreadsPerBlock();

  // Check for programmer-provided launch attribute...
  if (tpb > 0 && tpb <= maxThreadsPerBlock) {
    attrVal = std::string("1,") + utostr(TPB);
    kernelF->addFnAttr("amdgpu-flat-work-group-size", attrVal);

    if (clUseYLaunch)
      attrVal = std::string("1,") + utostr(TPB) + std::string(",1");
    else
      attrVal = utostr(TPB) + std::string(",1,1");
  } else if (defaultThreadsPerBlock > 0 &&
             defaultThreadsPerBlock <= maxThreadsPerBlock) {
    // Check for command line spec.
    attrVal = std::string("1,") + utostr(defaultThreadsPerBlock);
    kernelF->addFnAttr("amdgpu-flat-work-group-size", attrVal);

    if (clUseYLaunch)
      attrVal = std::string("1,") + utostr(defaultThreadsPerBlock)
                    + std::string(",1");
    else
      attrVal = utostr(defaultThreadsPerBlock) + std::string(",1,1");
  } else {
    // Use defaults...
    attrVal = std::string("1,") + utostr(maxThreadsPerBlock);
    kernelF->addFnAttr("amdgpu-flat-work-group-size", attrVal);

    if (clUseYLaunch)
      attrVal = std::string("1,") + utostr(maxThreadsPerBlock) +
                std::string(",1");
    else
      attrVal = utostr(maxThreadsPerBlock) + std::string(",1,1");
  }
  // Attribute falls through from above conditionals...
  kernelF->addFnAttr("amdgpu-max-num-workgroups", attrVal);
#endif

  // Tapir uses canonical induction variables in the range [0, end) with
  // stride 1. `end` is always the second parameter to the kernel function.
  Argument *end = kernelF->getArg(1);

  // Get the grainsize value, which is either constant or the third LC arg.
  // TODO: We only support a grain size of 1 right now. Not clear if this
  // could be a future optimization but strip mining on our current tests only
  // results in degraded performance.
  // if (unsigned gs = tl.getGrainsize())
  //  grainsize = ConstantInt::get(ivType, gs);
  // else
  Value *grainsize = ConstantInt::get(ivType, defaultGrainsize.getValue());

  IRBuilder<> builder(bbEntry->getTerminator());

  // Get the thread ID for this invocation of Helper.
  //
  // This is the classic thread ID calculation:
  //      i = blockDim.x * blockIdx.x + threadIdx.x;
  // For now we only generate 1-D thread IDs.
  Value *threadIdx;
  Value *blockDim;

  if (not clUseYLaunch) {
    Value *workItemIdX = emitWorkItemId(builder, 0);
    threadIdx = builder.CreateIntCast(workItemIdX, ivType,
                                      /*isSigned=*/false, ".kern.tidx.x");
    Value *workGroupSizeX = emitWorkGroupSize(builder, 0);
    blockDim = builder.CreateIntCast(workGroupSizeX, ivType,
                                     /*isSigned=*/false, ".kern.blkdim.x");
  } else {
    Value *workItemIdY = emitWorkItemId(builder, 1);
    threadIdx = builder.CreateIntCast(workItemIdY, ivType,
                                      /*isSigned=*/false, ".kern.tidx.y");
    Value *workGroupSizeY = emitWorkGroupSize(builder, 1);
    blockDim = builder.CreateIntCast(workGroupSizeY, ivType,
                                     /*isSigned=*/false, ".kern.blkdim.y");
  }

  Value *workGroupIdX = emitWorkGroupId(builder, 0);
  Value *blockIdx = builder.CreateIntCast(workGroupIdX, ivType,
                                          /*isSigned=*/false, ".kern.blkid.x");

  Value *blockOff = builder.CreateMul(blockIdx, blockDim, ".kern.blkoff.x");
  Value *threadID = builder.CreateAdd(threadIdx, blockOff, ".kern.tid");

  // threadID = Builder.CreateMul(threadID, grainsize);
  Value *threadEnd = builder.CreateAdd(threadID, grainsize, ".kern.last_idx");
  Value *cond = builder.CreateICmpUGE(threadID, end, ".kern.at_end");
  ReplaceInstWithInst(bbEntry->getTerminator(),
                      BranchInst::Create(bbExit, bbHeader, cond));

  // Replace the loop's induction variable with the GPU thread id.
  iv->getIncomingValueForBlock(bbEntry)->replaceAllUsesWith(threadID);

  // Update cloned loop condition to use the thread-end value.
  unsigned tripCountIdx = 0;
  ICmpInst *clonedCond = cast<ICmpInst>(vmap[tl.getCondition()]);
  if (clonedCond->getOperand(0) != end)
    ++tripCountIdx;
  assert(clonedCond->getOperand(tripCountIdx) == end &&
         "End argument not used in condition!");
  clonedCond->setOperand(tripCountIdx, threadEnd);
}

void HipLoop::processOutlinedLoopCall(TapirLoopInfo &tl, TaskOutlineInfo &toi,
                                      DominatorTree &dt) {
  LLVM_DEBUG(dbgs() << "hiploop: processing outlined loop call...\n"
                    << "\tkernel name: " << kernelName << "\n");

  LLVMContext &ctx = M.getContext();
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);

  Constant *zero = ConstantInt::get(i64, 0);
  ConstantInt *ctt = createConstInt(TTID::Hip, ctx);
  GlobalVariable *kProps =
      createKernelPropertiesGlobal(kernelName, TTID::Hip, M);
  Value *kName = createConstString(kernelName, M);
  GlobalVariable *embFB = getEmbFBGlobal(TTID::Hip, M);

  // At this point we need a threads-per-block value for the launch call. The
  // runtime will determine this value if ThreadsPerBlock is zero but it can
  // also be overridden via kitsune's forall launch attribute. The catch here is
  // the launch attribute's value for this is flexible and be a computed
  // expression vs. a compile-time constant. For this first step of creating the
  // kernel launch, we take the path of a runtime configuration vs. an
  // attributed launch.
  unsigned tpbHint = getTapirLoopThreadsPerBlockAttr(*tl.getLoop()).value_or(0);
  unsigned fixedTPB = getOptions().getFixedThreadsPerBlock();
  Value *tpb;
  if (tpbHint)
    tpb = ConstantInt::get(i32, tpbHint);
  else if (fixedTPB)
    tpb = ConstantInt::get(i32, fixedTPB);
  else
    tpb = ConstantInt::get(i32, 0);

  CallBase *callOutlined = cast<CallBase>(toi.ReplCall);
  BasicBlock *bbNew = callOutlined->getParent()->splitBasicBlock(callOutlined);
  IRBuilder<> builder(&bbNew->front());

  // Deal with type mismatches for the trip count.
  Value *tripCount = callOutlined->getArgOperand(1);
  if (tripCount->getType() != i64)
    tripCount = builder.CreateSExtOrBitCast(tripCount, i64, "cast.tc");

  // We need to explicitly sync non-const globals that are used in the kernel
  // before the kernel is launched.
  copyNonConstGlobalsHToD(usedGlobalValues, TTID::Hip, M, builder);

  Value *hipStream =
      builder.CreateIntrinsic(Intrinsic::kit_thread_stream, {ctt});
  SmallVector<Value *, 16> args = {
      ctt, embFB, kName, tripCount, zero, zero, tpb, kProps, hipStream,
  };
  for (Value *inp : callOutlined->args())
    args.push_back(inp);

  // TODO: We should probably have the launch and sync kitsune intrinsics take
  // a sync region as an argument This may make it easier to do post-outlining
  // analyses to eliminate/delay device synchronization calls instead of
  // always synchronizing immediately after the kernel launch.
  LLVM_DEBUG(dbgs() << "\t*- code gen kernel launch....\n");
  (void)builder.CreateIntrinsic(Intrinsic::kit_async_launch_kernel, args);
  (void)builder.CreateIntrinsic(Intrinsic::kit_sync_stream, {ctt, hipStream});

  // After the kernel is done, copy the non-const globals back to the host. This
  // is done here to keep this part of the code generation simple. A subsequent
  // pass will attempt to move this call to the point where the global is
  // actually used on the host (or perhaps even delete it if the host never uses
  // the global again).
  copyNonConstGlobalsDToH(usedGlobalValues, TTID::Hip, M, builder);

  callOutlined->eraseFromParent();
  LLVM_DEBUG(dbgs() << "*** finished processing outlined call.\n");
}

// As is the pattern with the GPU targets, the HipABI is setup to process all
// Tapir constructs within a given input Module (M). It then creates a
// corresponding module that contains the transformed device-side code. This is
// the kernelModule that is created below in the target constructor.
HipABI::HipABI(Module &hostM, const TTOptions &tto)
    : TapirTarget(hostM, tto), kernelModule("", hostM.getContext()),
      nextKernelID(0) {
  LLVM_DEBUG(dbgs() << "hipABI: HipABI::HipABI()\n");

  TargetMachine *tm = createTargetMachine(TTID::Hip, tto);
  kernelModule.setTargetTriple(tm->getTargetTriple());
  kernelModule.setDataLayout(tm->createDataLayout());

  kernelModule.setModuleIdentifier(getNameForDeviceModule(M, HIPABI_PREFIX));
  addDeviceModuleFlagsAttr(kernelModule, TTID::Hip);
  cloneModuleFlagsMetadataInto(M, kernelModule);
  cloneIdentMetadataInto(M, kernelModule);
}

HipABI::~HipABI() { /* no-op */ }

Value *HipABI::lowerGrainsizeCall(CallInst *grainsizeCall) {
  // TODO: The grain size on the GPU is a completely different beast than the
  // CPU cases Tapir was originally designed for. At present keeping the grain
  // size at 1 has almost always shown to yield the best results in terms of
  // performance but we should take a closer look...  We have some tweaks for
  // experimenting with this via the command line but it remains unexplored.
  Type *gsType = grainsizeCall->getType();
  Value *gs = ConstantInt::get(gsType, defaultGrainsize);

  grainsizeCall->replaceAllUsesWith(gs);
  grainsizeCall->eraseFromParent();
  return gs;
}

void HipABI::lowerSync(SyncInst &si) {
  // This tapir target splits the code into two modules, one for the host, the
  // other for the device. The sync instruction will only be present on the host
  // module.
}

void HipABI::preProcessModule() {
  // Create the global variable that will eventually contain the fat binary of
  // GPU code. This is currently uninitialized, but will be passed to several
  // of the kitsune runtime intrinsic calls when launching kernels, copying
  // global variables from host to device etc.
  (void)createEmbFBGlobal(TTID::Hip, M);
}

void HipABI::postProcessModule() {
  // At this point, we are done with the minimum task of outlining the tapir
  // loop into a kernel module. There are still a number of transformations that
  // must be carried out on this module before it can be compiled to GPU code,
  // but those will be done by subsequent passes. The module here is in a state
  // where we can perform combined host/device analyses and optimizations.
  (void)createEmbBCGlobal(kernelModule, TTID::Hip, M);
}

LoopOutlineProcessor *HipABI::getLoopOutlineProcessor(const TapirLoopInfo *tl) {
  LLVM_DEBUG(dbgs() << "hipabi: create loop outlining processor.\n");
  LLVM_DEBUG(saveModuleToFile(&M, M.getName().str() + ".input"));

  std::string kernelName =
      getNameForTapirLoop(*tl, HIPABI_KERNEL_NAME_PREFIX, nextKernelID++);
  return new HipLoop(M, kernelModule, kernelName, this->getOptions());
}
