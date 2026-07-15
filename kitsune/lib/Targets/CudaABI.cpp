//===- CudaABI.cpp - Tapir target for NVIDIA GPU's ------------------------===//
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
// Tapir target for NVIDIA GPU's.
//
// This tapir target outlines a tapir loop into a kernel function in a separate
// device module. This module will eventually be compiled to NVIDIA GPU code.
// Calls are added in the host module to launch these kernels. However, there is
// a lot more that that needs to be done before the device module can be
// compiled. Those steps are deferred to other passes that run later in the
// pipeline.
//
// NOTE: We currently do not support the full range of GPU architectures
// supported by the NVPTX backend. This is primarily due to a lack of resources
// to test every GPU.
//
// For some background material see the NVPTX target documentation
// at https://llvm.org/docs/NVPTXUsage.html.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Targets/CudaABI.h"
#include "GPUTTLoop.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/KernelProperties.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/TTOptions.h"
#include "kitsune/Core/ValueUtils.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "cuabi"

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

/// ptxas has several restrictions on the names of symbols, including internal
/// symbols. We modify the name of all symbols, including internal ones to be
/// valid C identifiers by replacing the invalid characters with an '_'.
/// However, there is a chance of collisions with other symbols in the module.
/// In most languages that we care about, a double-underscore at the start of an
/// identifier name is reserved for the compiler, so we prefix the newly created
/// names with a prefix starting with a double-underscore.
static std::string convertNameForPTX(StringRef name) {
  return normalizeSymbolName(name, "__kitcu__nwnm__");
}

/// Loop outline processor that transforms tapir loop nests into a kernel
/// functions for an NVIDIA GPU.
/// \ingroup kitsune
class CudaLoop : public GPUTTLoopBase {
protected:
  /// Set the correct attributes on the kernel function, \p f.
  virtual void setKernelFuncAttrs(Function &f) override;

  /// Set the correct calling convention on the kernel function, \p f.
  virtual void setKernelFuncCallingConv(Function &f) override;

  /// Set module-level attributes for the kernel function, \p f.
  virtual void setModuleAttrsForKernelFunc(Function &f) override;

public:
  CudaLoop(Module &hostM, Module &devM, const TTOptions &tto,
           const TapirLoopInfo &tl, StringRef kernelName);
  virtual ~CudaLoop() = default;

  /// Process the tapir loop \p tl jut before it is outlined. The \p vmap can
  /// be modified here for more control over how the outlining is performed.
  void preProcessTapirLoop(TapirLoopInfo &tl, ValueToValueMapTy &vmap) override;
};

CudaLoop::CudaLoop(Module &hostM, Module &devM, const TTOptions &tto,
                   const TapirLoopInfo &tl, StringRef kernelName)
    : GPUTTLoopBase(hostM, devM, tto, tl, TTID::Cuda, kernelName) {
  LLVM_DEBUG(dbgs() << "debug[cuabi]: creating a cuda loop outliner.\n"
                    << "  - target kernel name: " << kernelName << "\n");
}

void CudaLoop::preProcessTapirLoop(TapirLoopInfo &tl, ValueToValueMapTy &vmap) {
  GPUTTLoopBase::preProcessTapirLoop(tl, vmap);

  // ptxas imposes restrictions on the names that global entities may have.
  // Ideally, it would be good to do this in a post-processing pass, say the
  // prepare embedded module pass. However, the names of the globals must be
  // passed to Kitsune's intrinsics, so we have to do this here. The
  // alternative would involve an unhealthy amount of value chasing across two
  // different LLVM modules and is almost certainly not worth the trouble.
  for (GlobalValue *v : usedGlobalValues)
    if (auto *g = dyn_cast<GlobalVariable>(v))
      getDevGlobal(g, vmap)->setName(convertNameForPTX(g->getName()));
}

void CudaLoop::setKernelFuncAttrs(Function &f) {
  // Remove all target-related attributes from the kernel function. These may
  // be present because the frontend believes that the code is being compiled
  // for the CPU (host) only.
  f.removeFnAttr("target-cpu");
  f.removeFnAttr("target-features");
  f.removeFnAttr("tune-cpu");

  // Remove some functions that are relevant for functionality that is not
  // supported on the GPU. For instance, exceptions are not currently
  // available on GPU's.
  f.removeFnAttr("personality");

  // Replace some of the target-specific attributes with the correct ones.
  const TTOptions &tto = getOptions();
  StringRef arch = tto.getCudaArch();
  StringRef features = tto.getCudaTargetFeatures();

  f.addFnAttr("target-cpu", arch);
  f.addFnAttr("target-features", join_items(",", features, arch));

  // Add other attributes that are relevant for the target.
  f.addFnAttr("uniform-work-group-size", "true");
}

void CudaLoop::setModuleAttrsForKernelFunc(Function &f) {
  LLVMContext &ctx = f.getContext();
  Type *i32 = Type::getInt32Ty(ctx);

  SmallVector<Metadata *, 6> ops;
  ops.push_back(ValueAsMetadata::get(&f));
  ops.push_back(MDString::get(ctx, "kernel"));
  ops.push_back(ValueAsMetadata::get(ConstantInt::get(i32, 1)));
  // av.push_back(MDString::get(ctx, "maxntidx"));
  // av.push_back(ValueAsMetadata::get(
  //     ConstantInt::get(i32, MaxThreadsPerBlock)));

  NamedMDNode *annotations = devM.getOrInsertNamedMetadata("nvvm.annotations");
  annotations->addOperand(MDNode::get(ctx, ops));
}

void CudaLoop::setKernelFuncCallingConv(Function &f) {
  f.setCallingConv(CallingConv::PTX_Kernel);
}

CudaABI::CudaABI(Module &hostM, const TTOptions &tto)
    : GPUTTBase(TTID::Cuda, hostM, tto) {
  LLVM_DEBUG(dbgs() << "cuabi: CudaABI::CudaABI()\n");

  devM.setModuleFlag(Module::Override, "nvvm-reflect-ftz", clFTZ);
}

LoopOutlineProcessor *
CudaABI::getLoopOutlineProcessor(const TapirLoopInfo *tl) {
  LLVM_DEBUG(dbgs() << "cuabi: create loop outline processor.\n");
  LLVM_DEBUG(saveModuleToFile(&hostM, hostM.getName().str() + ".input"));

  const TTOptions &tto = getOptions();
  std::string kernelName = getNameForTapirLoop(*tl);

  return new CudaLoop(hostM, devM, tto, *tl, kernelName);
}
