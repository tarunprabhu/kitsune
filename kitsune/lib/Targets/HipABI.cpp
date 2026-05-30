//===- HipABI.cpp - Tapir target for AMD GPU's ----------------------------===//
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
// Tapir target for AMD GPU's.
//
// This tapir target outlines a tapir loop into a kernel function in a separate
// device module. This module will eventually be compiled to NVIDIA GPU code.
// Calls are added in the host module to launch these kernels. However, there is
// a lot more that that needs to be done before the device module can be
// compiled. Those steps are deferred to other passes that run later in the
// pipeline.
//
// NOTE: We currently do not support the full range of GPU architectures
// supported by the AMDGPU backend. This is primarily due to a lack of resources
// to test every GPU.
//
// For some background material see the AMDGPU target documentation
// at: https://llvm.org/docs/AMDGPUUsage.html
//
//===----------------------------------------------------------------------===//

#include "kitsune/Targets/HipABI.h"
#include "GPUTTLoop.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/InstUtils.h"
#include "kitsune/Core/KernelProperties.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/TTOptions.h"
#include "kitsune/Core/ValueUtils.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "hipabi"

// FIXME: We really should not be exposing command line options from other
// source files. This is an experimental option that has been hacked in for the
// moment. If this is useful, we should consider adding it to the tapir target
// options instead. Otherwise, it should be removed altogether.
cl::opt<bool> clUseYLaunch("hipabi-y-launch", cl::init(false), cl::Hidden,
                           cl::desc("Launch kernel using y-axis threading"),
                           cl::cat(cl::catKitClDevOpts));

/// Loop outline processor that transforms tapir loop nests into a kernel
/// functions for an AMD GPU.
/// \ingroup kitsune
class HipLoop : public GPUTTLoopBase {
protected:
  /// Get the address space, in the derived module, for constant global
  /// variables.
  virtual unsigned getConstAddrSpace() const override;

  /// get the address space, in the derive module, for non-constant global
  /// variables.
  virtual unsigned getNonConstAddrSpace() const override;

  /// Set the correct attributes on the kernel function \p f.
  virtual void setKernelFuncAttrs(Function &f) override;

  /// Set the correct calling convention on the kernel function \p f.
  virtual void setKernelFuncCallingConv(Function &f) override;

  /// Set the correct visibility on the kernel function \p f.
  virtual void setKernelFuncVisibility(Function &f) override;

public:
  HipLoop(Module &hostM, Module &devM, const TTOptions &tto,
          const TapirLoopInfo &tl, StringRef kernelName);
  virtual ~HipLoop() = default;

  /// Process the tapir loop \p tl jut before it is outlined. The \p vmap can
  /// be modified here for more control over how the outlining is performed.
  void preProcessTapirLoop(TapirLoopInfo &tl, ValueToValueMapTy &vmap) override;
};

HipLoop::HipLoop(Module &hostM, Module &devM, const TTOptions &tto,
                 const TapirLoopInfo &tl, StringRef kernelName)
    : GPUTTLoopBase(hostM, devM, tto, tl, TTID::Hip, kernelName) {
  LLVM_DEBUG(
      dbgs() << "hipabi: hip loop outliner creation:\n"
             << "\ttransforming loop to kernel: " << kernelName << "(...)\n"
             << "\tdevice-side module name    : " << devM.getName() << "\n\n");
}

unsigned HipLoop::getConstAddrSpace() const {
  return AMDGPUAS::CONSTANT_ADDRESS;
}

unsigned HipLoop::getNonConstAddrSpace() const {
  return AMDGPUAS::GLOBAL_ADDRESS;
}

void HipLoop::preProcessTapirLoop(TapirLoopInfo &tl, ValueToValueMapTy &vmap) {
  GPUTTLoopBase::preProcessTapirLoop(tl, vmap);

  // HIP appears to require protected visibility. Without this, attempting to
  // link the fat binary results in a relocation error.
  for (GlobalValue *v : usedGlobalValues)
    if (auto *g = dyn_cast<GlobalVariable>(v))
      if (!g->isConstant())
        getDevGlobal(g, vmap)->setVisibility(GlobalValue::ProtectedVisibility);
}

void HipLoop::setKernelFuncAttrs(Function &f) {
  // Remove any attributes that are only relevant for the host.
  f.removeFnAttr("target-cpu");
  f.removeFnAttr("target-features");
  f.removeFnAttr("tune-cpu");

  // Remove other attributes that we cannot deal with in any reasonable way in
  // the device
  f.removeFnAttr(Attribute::UWTable);

  // Add new target-specific attributes
  f.addFnAttr("target-cpu", getOptions().getHipArch());
  f.addFnAttr("target-features", getOptions().getHipTargetFeatures());

  // Add other attributes that are either required or desirable.
  f.addFnAttr("no-trapping-math", "true");
  f.addFnAttr(Attribute::MustProgress);
  f.addFnAttr(Attribute::NoUnwind);

  // This only works when the code object version >= 5, but we have ensured that
  // this is the case in the frontend.
  f.addFnAttr("uniform-work-group-size", "true");

  // Specify the minimum and maximum flat work group sizes that will be used
  // when the kernel is dispatched.
  std::string attrVal = "128,1024";
  f.addFnAttr("amdgpu-flat-work-group-size", attrVal);

  // FIXME: It is not clear why we are still carrying this around.
#if 0 // DISABLED FOR TESTING
  unsigned maxThreadsPerBlock = getOptions().getMaxThreadsPerBlock();
  unsigned defaultThreadsPerBlock = getOptions().getFixedThreadsPerBlock();

  // Check for programmer-provided launch attribute...
  if (tpb > 0 && tpb <= maxThreadsPerBlock) {
    attrVal = std::string("1,") + utostr(TPB);
    f.addFnAttr("amdgpu-flat-work-group-size", attrVal);

    if (clUseYLaunch)
      attrVal = std::string("1,") + utostr(TPB) + std::string(",1");
    else
      attrVal = utostr(TPB) + std::string(",1,1");
  } else if (defaultThreadsPerBlock > 0 &&
             defaultThreadsPerBlock <= maxThreadsPerBlock) {
    // Check for command line spec.
    attrVal = std::string("1,") + utostr(defaultThreadsPerBlock);
    f.addFnAttr("amdgpu-flat-work-group-size", attrVal);

    if (clUseYLaunch)
      attrVal = std::string("1,") + utostr(defaultThreadsPerBlock)
                    + std::string(",1");
    else
      attrVal = utostr(defaultThreadsPerBlock) + std::string(",1,1");
  } else {
    // Use defaults...
    attrVal = std::string("1,") + utostr(maxThreadsPerBlock);
    f.addFnAttr("amdgpu-flat-work-group-size", attrVal);

    if (clUseYLaunch)
      attrVal = std::string("1,") + utostr(maxThreadsPerBlock) +
                std::string(",1");
    else
      attrVal = utostr(maxThreadsPerBlock) + std::string(",1,1");
  }

  // Attribute falls through from above conditionals...
  f.addFnAttr("amdgpu-max-num-workgroups", attrVal);
#endif
}

void HipLoop::setKernelFuncCallingConv(Function &f) {
  f.setCallingConv(CallingConv::AMDGPU_KERNEL);
}

void HipLoop::setKernelFuncVisibility(Function &f) {
  // AMD's runtime requires that the kernel function have protected visibility.
  // Otherwise, it is unable to find the kernel function at run-time. This, in
  // turn, requires the function to have external linkage. This is the linkage
  // set on kernel functions by default.
  f.setVisibility(GlobalValue::ProtectedVisibility);
}

// The loop output processor that performs Y-axis launches for 1D loops.
//
// This is only here because we want the default loop outline processors to be
// clean. This isolates some code that was present in the original
// implementation, but where it wasn't clear if there was any advantage to
// having it around.
//
// FIXME: We should consider if there is any advantage to having this.
class HipLoop1Y : public HipLoop {
protected:
  virtual void processOutlinedIVs(Function &f, TapirLoopInfo &tl,
                                  const ValueToValueMapTy &vmap) override {
    assert(getDepth() == 1 &&
           "Y-axis launches are only supported on loops with depth 1");

    LLVMContext &ctx = f.getContext();
    Loop *loop = tl.getLoop();

    auto *bbEntry = cast<BasicBlock>(vmap.lookup(loop->getLoopPreheader()));
    auto *bbHeader = cast<BasicBlock>(vmap.lookup(loop->getHeader()));
    auto *bbExit = cast<BasicBlock>(vmap.lookup(tl.getExitBlock()));
    auto *iv = cast<PHINode>(vmap.lookup(tl.getPrimaryInduction().first));
    Type *ivType = iv->getType();

    IRBuilder<> bldr(bbEntry->getTerminator());

    Value *ctt = toConstant(tt, ctx);
    Value *tidX = bldr.CreateIntrinsic(Intrinsic::kit_gpu_thread_id_y, {ctt},
                                       /*FMFSource=*/{}, "tid.x");
    Value *bidX = bldr.CreateIntrinsic(Intrinsic::kit_gpu_block_id_x, {ctt},
                                       /*FMFSource=*/{}, "bid.x");
    Value *bszX = bldr.CreateIntrinsic(Intrinsic::kit_gpu_block_size_y, {ctt},
                                       /*FMFSource=*/{}, "bsz.x");
    Value *bdxbi = bldr.CreateMul(bszX, bidX);
    Value *tipbdxbi = bldr.CreateAdd(bdxbi, tidX, ".ivbeg.x");
    Value *ivBeg =
        bldr.CreateIntCast(tipbdxbi, ivType, /*isSigned=*/false, "ivbeg.x");
    Value *grainsize = getGrainsize(ivType);
    Value *ivEnd = bldr.CreateAdd(ivBeg, grainsize, "ivend.x");

    Argument *tcX = f.getArg(1);
    Value *ivCond = bldr.CreateICmpUGE(ivBeg, tcX);
    ReplaceInstWithInst(bbEntry->getTerminator(),
                        BranchInst::Create(bbExit, bbHeader, ivCond));

    iv->getIncomingValueForBlock(bbEntry)->replaceAllUsesWith(ivBeg);
    updateTripCount(loop, iv, ivEnd, vmap);
  }

public:
  HipLoop1Y(Module &hostM, Module &devM, const TTOptions &tto,
            const TapirLoopInfo &tl, StringRef kernelName)
      : HipLoop(hostM, devM, tto, tl, kernelName) {}

  virtual ~HipLoop1Y() = default;
};

// As is the pattern with the GPU targets, the HipABI is setup to process all
// Tapir constructs within a given input Module (M). It then creates a
// corresponding module that contains the transformed device-side code. This is
// the kernelModule that is created below in the target constructor.
HipABI::HipABI(Module &hostM, const TTOptions &tto)
    : GPUTTBase(TTID::Hip, hostM, tto) {
  LLVM_DEBUG(dbgs() << "hipabi: HipABI::HipABI()\n");
}

LoopOutlineProcessor *HipABI::getLoopOutlineProcessor(const TapirLoopInfo *tl) {
  LLVM_DEBUG(dbgs() << "hipabi: create loop outline processor.\n");
  LLVM_DEBUG(saveModuleToFile(&hostM, hostM.getName().str() + ".input"));

  const TTOptions &tto = getOptions();
  unsigned depth = getPerfectDepthAttr(*tl->getLoop()).value_or(0);
  std::string kernelName = getNameForTapirLoop(*tl);

  if (clUseYLaunch && depth > 1)
    llvm_unreachable(
        "Y-axis launches are only allowed with loop nests of depth 1");

  if (clUseYLaunch)
    return new HipLoop1Y(hostM, devM, tto, *tl, kernelName);
  return new HipLoop(hostM, devM, tto, *tl, kernelName);
}
