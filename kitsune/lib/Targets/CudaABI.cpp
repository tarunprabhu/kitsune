//===- CudaABI.cpp - Lower Tapir Kitsune's cuda runtime -----------------*-===//
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

#include "kitsune/Targets/CudaABI.h"
#include "GPUTTUtils.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/KernelProperties.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/TTOptions.h"
#include "kitsune/Core/TargetUtils.h"
#include "kitsune/Core/ValueUtils.h"
#include "kitsune/Frontend/CommandLineOptions.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "cuabi"

// For some background material see the NVPTX target documentation
// at https://llvm.org/docs/NVPTXUsage.html.
//
// NOTE: We currently do not support the full range of GPU architectures
// supported by the NVPTX backend. This is primarily due to a lack of resources
// to test every GPU.
//
// This transformation outlines a tapir loop into a kernel module that will
// eventually be compiled to NVIDIA GPU code. Calls are added in the original
// module to use Kitsune's cuda runtime to launch the tapir loops. There is a
// lot more that needs to be done before the kernel module can be compiled to
// GPU code, but those steps are handled in subsequent passes.

// This is meant to be a factor used for additional kernel optimizations but is
// currently not used this. It should be left in its default state.
static cl::opt<unsigned> defaultGrainsize(
    "cuabi-default-grainsize", cl::init(1), cl::Hidden,
    cl::desc("The default grain size used by the transform "
             "when analysis fails to determine one (default=1)"),
    cl::cat(cl::catKitClDevOpts));

// Enable/Disable flush denorms-to-zero code generation.
static cl::opt<bool> clFTZ("cuabi-ftz", cl::init(false), cl::Hidden,
                           cl::desc("Enable flush-denorms-to-zero"),
                           cl::cat(cl::catKitClDevOpts));

// FIXME: The default is currently set to true. This should be changed to false
// and the name of the option changed.
//
// FIXME: We really should not be exposing command line options from other
// source files. This is an experimental option that has been hacked in for the
// moment. If this is useful, we should consider adding it to the tapir target
// options instead. Otherwise, it should be removed altogether.
//
// Request that the runtime carry out an extra set of steps to attempt to refine
// the launch parameters of kernels. In this mode of operation the compiler will
// provide some compile-time information to the runtime for assisting in the
// assisting in the analysis and refinement of launches.
cl::opt<bool> clRefineLaunches(
    "cuabi-refine-launches", cl::init(true), cl::Hidden,
    cl::desc("Enable runtime's refinement of launch parameters"),
    cl::cat(cl::catKitClDevOpts));

/// This prefix is intentionally *NOT* __kitcuda to ensure that there is no
/// confusion - and, more importantly, no collisions - between any names
/// prefixed with this and the symbols from kitsune's cuda runtime which are
/// typically prefixed with __kitcuda.
static constexpr StringRef CUABI_PREFIX = "__kitcu_";
static constexpr StringRef CUABI_KERNEL_NAME_PREFIX = "__kitcu_loop_";

/// ptxas has several restrictions on the names of symbols, including internal
/// symbols. If the given name is not valid for PTX, return a modified name.
/// Otherwise, just return a clone of the name. The result is prefixed with a
/// string to reduce the likelihood of collisions. This behavior can be
/// overridden by passing false to \ref addPrefix.
static std::string convertNameForPTX(StringRef name, bool addPrefix = true) {
  auto isInvalidChar = [](char c) -> bool { return c == '.' or c == '-'; };
  if (std::none_of(name.begin(), name.end(), isInvalidChar))
    return name.str();

  // Simply replacing the invalid characters with _ may not be safe because
  // there is a chance of collisions with other symbols in the module. In most
  // languages that we care about, a double-underscore at the start of an
  // identifier name is reserved for the compiler, so we prefix the newly
  // created names with such a prefix.
  std::string buf;
  llvm::raw_string_ostream os(buf);
  if (addPrefix)
    os << CUABI_PREFIX << "_nwnm__";
  for (char c : name)
    os << (isInvalidChar(c) ? '_' : c);
  return buf;
}

/// The loop outline process for transforming a Tapir parallel loop into a
/// cuda kernel function.
/// \ingroup kitsune
class CudaLoop : public LoopOutlineProcessor {
private:
  /// The name of the kernel into which the loop is outlined.
  std::string kernelName;

  /// For GPU targets, we outline the loop into a separate module. This is that
  /// module.
  Module &kernelModule;

  /// The GlobalValue's used in the loop that is being outlined. This includes
  /// functions, global variables, aliases and ifunc's.
  SmallSet<GlobalValue *, 8> usedGlobalValues;

public:
  CudaLoop(Module &hostM, Module &kernelModule, const std::string &kernelName,
           const TTOptions &tto);
  ~CudaLoop();

  /// Setup the loop-control arguments \p lcArgs and loop-control inputs
  /// \p lcInputs for the Tapir loop \p tl.
  void setupLoopControlArgs(TapirLoopInfo *tl, SmallVectorImpl<Value *> &lcArgs,
                            SmallVectorImpl<Value *> &lcInputs) override;

  void preProcessTapirLoop(TapirLoopInfo &tl, ValueToValueMapTy &vmap) override;
  void postProcessOutline(TapirLoopInfo &tl, TaskOutlineInfo &toi,
                          ValueToValueMapTy &vmap) override;
  void processOutlinedLoopCall(TapirLoopInfo &tl, TaskOutlineInfo &toi,
                               DominatorTree &dt) override;
};

CudaLoop::CudaLoop(Module &hostM, Module &kernelModule,
                   const std::string &kernelName, const TTOptions &tto)
    : LoopOutlineProcessor(hostM, kernelModule, tto,
                           CloneFunctionChangeType::DifferentModule),
      kernelName(kernelName), kernelModule(kernelModule) {
  LLVM_DEBUG(dbgs() << "debug[cuabi]: creating a cuda loop outliner.\n"
                    << "  - target kernel name: " << kernelName << "\n");
}

CudaLoop::~CudaLoop() {
  LLVM_DEBUG(dbgs() << "debug[cuabi]: destroying loop outliner for kernel '"
                    << kernelName << "'.\n");
}

void CudaLoop::setupLoopControlArgs(TapirLoopInfo *tl,
                                    SmallVectorImpl<Value *> &lcArgs,
                                    SmallVectorImpl<Value *> &lcInputs) {
  InductionDescriptor ivDescr = tl->getPrimaryInduction().second;

  // It is not clear if we actually need the step value to be 1, but until we
  // can be sure of it, we'll be conservative and require it here.
  assert(ivDescr.getStep()->isOne() &&
         "Step of tapir loop induction variable must be 1");

  // We require tapir loops to be lowered to the GPU to have canonical
  // induction variables. This should have been checked before we get here, but
  // make sure that is the case.
  Value *ivBeg = ivDescr.getStartValue();
  assert(isZero(ivBeg) &&
         "Start value of tapir loop induction variable must be 0");

  Value *tc = tl->getTripCount();
  assert(tc && "No trip count found for Tapir loop end argument.");

  // Since the start value is 0, we don't strictly need this. However, not
  // passing this causes issues in loop spawning since that assumes that this
  // value will be passed. The fixes needed to make this work in loop spawning
  // are not particularly difficult, but it does feel messy. For now, we just
  // pass it since the fix to loop spawning will likely require some more
  // thought.
  LoopCtlArgs.push_back(new Argument(ivBeg->getType(), "iv0.x"));
  lcArgs.push_back(LoopCtlArgs.back());
  lcInputs.push_back(ivBeg);

  LoopCtlArgs.push_back(new Argument(tc->getType(), "tc.x"));
  lcArgs.push_back(LoopCtlArgs.back());
  lcInputs.push_back(tc);
}

void CudaLoop::preProcessTapirLoop(TapirLoopInfo &tl, ValueToValueMapTy &vmap) {
  LLVM_DEBUG(dbgs() << "debug[cuabi]: -preprocessing loop for kernel '"
                    << kernelName << "'.\n");

  // Collect the top-level entities (Function, GlobalVariable, GlobalAlias
  // and GlobalIFunc) that are used in the outlined loop. Since the outlined
  // loop will live in the kernelModule, any GlobalValue's used in it must be
  // be cloned into the kernelModule and then registered with the cuda
  // runtime. The registration will be done in the global ctor which will be
  // generated by a later pass.
  collectGlobalValues(*tl.getLoop(), usedGlobalValues);

  // NVPTX has a number of different address spaces. We do not use them and
  // the code seems to work. It is not clear if there is any advantage to
  // using them, but it may be a good idea to look into it at some point.
  cloneUsedGlobalVariablesInto(kernelModule, usedGlobalValues, vmap);

  // ptxas imposes restrictions on the names that global entities may have.
  // Ideally, it would be good to do this in a post-processing pass, say the
  // prepare embedded module pass. However, the names of the globals must be
  // passed to Kitsune's intrinsics, so we have to do this here. The
  // alternative would involve an unhealthy amount of value chasing across two
  // different LLVM modules and is almost certainly not worth the trouble.
  for (GlobalValue *v : usedGlobalValues)
    if (auto *g = dyn_cast<GlobalVariable>(v))
      cast<GlobalVariable>(vmap[g])->setName(convertNameForPTX(g->getName()));

  // The global variables have to be cloned before cloning the functions
  // because they may be used in the bodies of functions to be cloned.
  cloneReachableFuncsInto(kernelModule, usedGlobalValues, vmap);
  cloneReachableIFuncsInto(kernelModule, usedGlobalValues, vmap);

  // The aliasee in global aliases is a global value, so they must be cloned
  // after the global variables and functions are in the vmap.
  cloneUsedGlobalAliasesInto(kernelModule, usedGlobalValues, vmap);
}

void CudaLoop::postProcessOutline(TapirLoopInfo &tl, TaskOutlineInfo &toi,
                                  ValueToValueMapTy &vmap) {
  LLVMContext &ctx = M.getContext();
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

  Function *kernelF = toi.Outline;

  // Set the generated name for the kernel. This name is passed to the
  // runtime's kernel launch function, so it must be set correctly.
  kernelF->setName(kernelName);

  // Set the linkage of the kernel to external to prevent it from being DCE'ed
  // since there will be no caller for the function in the kernel module.
  kernelF->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
  kernelF->setCallingConv(CallingConv::PTX_Kernel);

  // Remove all target-related attributes from the kernel function. These may
  // be present because the frontend believes that the code is being compiled
  // for the CPU (host) only.
  kernelF->removeFnAttr("target-cpu");
  kernelF->removeFnAttr("target-features");
  kernelF->removeFnAttr("tune-cpu");

  // Remove some functions that are relevant for functionality that is not
  // supported on the GPU. For instance, exceptions are not currently
  // available on GPU's.
  kernelF->removeFnAttr("personality");

  // Add an attribute identifying this as a function outlined from a tapir
  // loop.
  kernelF->addFnAttr(Attribute::KitKernel);

  // Replace some of the target-specific attributes with the correct ones.
  kernelF->addFnAttr("target-cpu", getOptions().getCudaArch());
  kernelF->addFnAttr("target-features",
                     join_items(",", getOptions().getCudaTargetFeatures(),
                                getOptions().getCudaArch()));

  // Add other attributes that are relevant for the target.
  kernelF->addFnAttr("uniform-work-group-size", "true");

  NamedMDNode *annotations =
      kernelModule.getOrInsertNamedMetadata("nvvm.annotations");
  SmallVector<Metadata *, 6> av;
  av.push_back(ValueAsMetadata::get(kernelF));
  av.push_back(MDString::get(ctx, "kernel"));
  av.push_back(
      ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(ctx), 1)));
  // av.push_back(MDString::get(ctx, "maxntidx"));
  // av.push_back(ValueAsMetadata::get(
  //     ConstantInt::get(Type::getInt32Ty(ctx), MaxThreadsPerBlock)));
  annotations->addOperand(MDNode::get(ctx, av));

  // Tapir uses canonical induction variables in the range [0, end) with
  // stride 1. `end` is always the second parameter to the kernel function.
  Argument *tcX = kernelF->getArg(1);

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
  // This is the classic CUDA thread ID calculation:
  //      i = blockDim.x * blockIdx.x + threadIdx.x;
  // For now we only generate 1-D thread IDs.
  Value *threadIdx =
      builder.CreateIntrinsic(Intrinsic::kit_gpu_thread_id_x, {}, {}, "tid.x");
  Value *blockIdx =
      builder.CreateIntrinsic(Intrinsic::kit_gpu_block_id_x, {}, {}, "bid.x");
  Value *blockDim =
      builder.CreateIntrinsic(Intrinsic::kit_gpu_block_size_x, {}, {}, "bsz.x");
  Value *bdxbi = builder.CreateMul(blockIdx, blockDim);
  Value *tipbdxbi = builder.CreateAdd(threadIdx, bdxbi, ".ivbeg.x");
  Value *ivBeg =
      builder.CreateIntCast(tipbdxbi, ivType, /*isSigned=*/false, "ivbeg.x");

  // threadID = builder.CreateMul(ThreadID, Grainsize);
  Value *ivEnd = builder.CreateAdd(ivBeg, grainsize, "ivend.x");
  Value *ivCond = builder.CreateICmpUGE(ivBeg, tcX);
  ReplaceInstWithInst(bbEntry->getTerminator(),
                      BranchInst::Create(bbExit, bbHeader, ivCond));

  // Use the thread ID as the start iteration number for the primary IV.
  iv->getIncomingValueForBlock(bbEntry)->replaceAllUsesWith(ivBeg);
  // TODO: ???? PrimaryIVInput->eraseFromParent();

  // Update cloned loop condition to use the thread-end value.
  unsigned tripCountIdx = 0;
  ICmpInst *clonedCond = cast<ICmpInst>(vmap[tl.getCondition()]);
  if (clonedCond->getOperand(0) != tcX)
    ++tripCountIdx;
  assert(clonedCond->getOperand(tripCountIdx) == tcX &&
         "End argument not used in condition!");
  clonedCond->setOperand(tripCountIdx, ivEnd);
}

void CudaLoop::processOutlinedLoopCall(TapirLoopInfo &tl, TaskOutlineInfo &toi,
                                       DominatorTree &dt) {
  LLVM_DEBUG(dbgs() << "cudaloop: processing outlined loop call...\n"
                    << "\tkernel name: " << kernelName << "\n");

  LLVMContext &ctx = M.getContext();
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);

  Constant *zero = ConstantInt::get(i64, 0);
  Constant *ctt = toConstant(TTID::Cuda, ctx);
  GlobalVariable *kprops =
      createKernelPropertiesGlobal(kernelName, TTID::Cuda, M);
  Value *kName = createConstString(kernelName, M);
  GlobalVariable *embFB = getEmbFBGlobal(TTID::Cuda, M);

  // At this point we need a threads-per-block value for the launch call. The
  // runtime will determine this value if ThreadsPerBlock is zero but it can
  // also be overridden via kitsune's forall launch attribute. The catch here
  // is the launch attribute's value for this is flexible and be a computed
  // expression vs. a compile-time constant. For this first step of creating
  // the kernel launch, we take the path of a runtime configuration vs. an
  // attributed launch.
  unsigned tpbHint = getThreadsPerBlockAttr(*tl.getLoop()).value_or(0);
  Value *tpb = ConstantInt::get(i32, tpbHint);

  CallBase *callOutlined = cast<CallBase>(toi.ReplCall);
  BasicBlock *bbNew = callOutlined->getParent()->splitBasicBlock(callOutlined);
  IRBuilder<> builder(&bbNew->front());

  // Deal with type mismatches for the trip count.
  Value *tripCount = callOutlined->getArgOperand(1);
  if (tripCount->getType() != i64)
    tripCount = builder.CreateSExtOrBitCast(tripCount, i64, "cast.tc");

  // We need to explicitly sync non-const globals that are used in the kernel
  // before the kernel is launched.
  copyNonConstGlobalsHToD(usedGlobalValues, TTID::Cuda, M, builder);

  Value *cudaStream =
      builder.CreateIntrinsic(Intrinsic::kit_thread_stream, {ctt});
  SmallVector<Value *, 16> args = {
      ctt, embFB, kName, tripCount, zero, zero, tpb, kprops, cudaStream,
  };
  for (Value *inp : callOutlined->args())
    args.push_back(inp);

  // TODO: We should probably have the launch and sync kitsune intrinsics take
  // a sync region as an argument This may make it easier to do post-outlining
  // analyses to eliminate/delay device synchronization calls instead of
  // always synchronizing immediately after the kernel launch.
  LLVM_DEBUG(dbgs() << "\t*- code gen kernel launch....\n");
  (void)builder.CreateIntrinsic(Intrinsic::kit_async_launch_kernel, args);
  (void)builder.CreateIntrinsic(Intrinsic::kit_sync_stream, {ctt, cudaStream});

  // After the kernel is done, copy the non-const globals back to the host.
  // This is done here to keep this part of the code generation simple. A
  // subsequent pass will attempt to move this call to the point where the
  // globals are actually used on the host (or perhaps even delete it if the
  // host never uses the global again).
  copyNonConstGlobalsDToH(usedGlobalValues, TTID::Cuda, M, builder);

  callOutlined->eraseFromParent();
  LLVM_DEBUG(dbgs() << "*** finished processing outlined call.\n");
}

CudaABI::CudaABI(Module &hostM, const TTOptions &tto)
    : TapirTarget(hostM, tto), kernelModule("", hostM.getContext()),
      nextKernelID(0) {
  LLVM_DEBUG(dbgs() << "cuabi: CudaABI::CudaABI()\n");

  TargetMachine *tm = createTargetMachine(TTID::Cuda, tto);
  kernelModule.setTargetTriple(tm->getTargetTriple());
  kernelModule.setDataLayout(tm->createDataLayout());

  kernelModule.setModuleIdentifier(getNameForDeviceModule(M, CUABI_PREFIX));
  addDeviceModuleFlagsAttr(kernelModule, TTID::Cuda);
  cloneModuleFlagsMetadataInto(M, kernelModule);
  cloneIdentMetadataInto(M, kernelModule);
  kernelModule.setModuleFlag(Module::Override, "nvvm-reflect-ftz", clFTZ);
}

CudaABI::~CudaABI() { LLVM_DEBUG(dbgs() << "cuabi: destroy tapir target.\n"); }

Value *CudaABI::lowerGrainsizeCall(CallInst *grainsizeCall) {
  // TODO: The grainsize on the GPU is a completely different beast than the CPU
  // cases Tapir was originally designed for. At present keeping the grainsize
  // at 1 has almost always shown to yield the best results.  It is obviously
  // not the best choice for all cases...
  Type *gsType = grainsizeCall->getType();
  Value *gs = ConstantInt::get(gsType, defaultGrainsize.getValue());

  grainsizeCall->replaceAllUsesWith(gs);
  grainsizeCall->eraseFromParent();
  return gs;
}

void CudaABI::lowerSync(SyncInst &si) {
  // This tapir target splits the code into two modules, one for the host, the
  // other for the device. The sync instruction will only be present on the host
  // module.
}

void CudaABI::preProcessModule() {
  // Create the global variable that will eventually contain the fat binary of
  // GPU code. This is currently uninitialized, but will be passed to several
  // of the kitsune runtime intrinsic calls when launching kernels, copying
  // global variables from host to device etc.
  (void)createEmbFBGlobal(TTID::Cuda, M);
}

void CudaABI::postProcessModule() {
  LLVM_DEBUG(dbgs() << "cuabi: post processing kernel and host modules...\n");

  // At this point, we are done with the minimum task of outlining the tapir
  // loop into a kernel module. There are still a number of transformations that
  // must be carried out on this module before it can be compiled to GPU code,
  // but those will be done by subsequent passes. The module here is in a state
  // where we can perform combined host/device analyses and optimizations.
  (void)createEmbBCGlobal(kernelModule, TTID::Cuda, M);
}

LoopOutlineProcessor *
CudaABI::getLoopOutlineProcessor(const TapirLoopInfo *tl) {
  LLVM_DEBUG(dbgs() << "cuabi: create loop outlining processor.\n");
  LLVM_DEBUG(saveModuleToFile(&M, M.getName().str() + ".input"));

  std::string kernelName = convertNameForPTX(
      getNameForTapirLoop(*tl, CUABI_KERNEL_NAME_PREFIX, nextKernelID++),
      /*AddPrefix=*/false);
  return new CudaLoop(M, kernelModule, kernelName, this->getOptions());
}
