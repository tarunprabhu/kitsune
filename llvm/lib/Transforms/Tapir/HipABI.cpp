//===- HipABI.cpp - Tapir to Kitsune runtime HIP target ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
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
//===----------------------------------------------------------------------===
//
// This file implements the Kitsune+Tapir HIP ABI to convert Tapir
// instructions to calls into the HIP-centric portions of the Kitsune
// runtime for HIP to produce a fully compiled fat binary inserted into
// the input LLVM Module.
//
// TODO: device-side calls to cover feature set and double-precision support
// TODO: add printf() support.
// TODO: revisit/refactor 'mutate' type uses.
// TODO: -- math options for:
//             - DAZ [on|off],
//             - unsafe math [on|off],
//             - sqrt rounding [on|off],
//             - etc.
// TODO: more robust target architecture processing
// TODO: better optimization and code gen.
//
//===----------------------------------------------------------------------===//
#include "llvm/Transforms/Tapir/HipABI.h"
#include "kitsune/Config/config.h"
#include "llvm-c/Core.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/TargetParser.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Tapir/TapirGPUUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Tapir/TapirStringUtils.h"
#include "llvm/Transforms/Utils/AMDGPUEmitPrintf.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "hipabi"

// For some background material see the AMDGPU target documentation
// at: https://llvm.org/docs/AMDGPUUsage.html
//
// This transformation is carrying out the prep to convert Tapir to a
// kernel module suitable for codegen using the AMDGPU target.

static cl::opt<std::string> clGPUArch(
    "tapir-hip-arch", cl::init(KITSUNE_HIP_ARCH_DEFAULT), cl::NotHidden,
    cl::desc("Target AMD GPU architecture (default = " KITSUNE_HIP_ARCH_DEFAULT
             ")"));

static cl::opt<int>
    OptLevel("hipabi-opt-level", cl::init(-1), cl::NotHidden,
             cl::desc("The Tapir HIP target transform optimization level"));

static cl::opt<unsigned> HostOptLevel( // EXPERIMENTAL
    "hipabi-host-opt-level", cl::init(0), cl::NotHidden,
    cl::desc("The optimization level for a final pass over the transformed "
             "host-side code."));

static cl::opt<bool>
    CodeGenPrefetch("hipabi-prefetch", cl::init(true), cl::Hidden,
                    cl::desc("Enable generation of calls to do data "
                             "prefetching for managed memory."));

enum ROCmABIVersion {
  ROCm_ABI_V4, // old
  ROCm_ABI_V5, // default
};

static cl::opt<ROCmABIVersion> ROCmABITarget(
    "hipabi-rocm-abi", cl::init(ROCm_ABI_V5), cl::Hidden,
    cl::desc("Select the targeted ROCm ABI version."),
    cl::values(clEnumValN(ROCm_ABI_V4, "v4", "Target ROCm v. 4 ABI.")),
    cl::values(clEnumValN(ROCm_ABI_V5, "v5", "Target ROCm v. 5 ABI.")));

static cl::opt<bool> Use64ElementWavefront(
    "hipabi-wavefront64", cl::init(true), cl::Hidden,
    cl::desc("Use 64 element wavefronts. (default: enabled)"));

static cl::opt<bool>
    EnableXnack("hipabi-xnack", cl::init(true), cl::NotHidden,
                cl::desc("Enable/disable xnack. (default: true)"));

static cl::opt<bool>
    EnableSRAMECC("hipabi-sramecc", cl::init(true), cl::NotHidden,
                  cl::desc("Enable/disable sramecc.(default: false)"));

static cl::opt<unsigned> DefaultGrainSize(
    "hipabi-default-grainsize", cl::init(1), cl::Hidden,
    cl::desc("The default grain size used by the transform "
             "when analysis fails to determine one. (default=1)"));

static cl::opt<bool>
    KeepIntermediateFiles("hipabi-keep-files", cl::init(false), cl::Hidden,
                          cl::desc("Keep/create intermediate files during the "
                                   "various stages of the transform."));

static cl::opt<bool>
    UseYLaunch("hipabi-y-launch", cl::init(false), cl::Hidden,
               cl::desc("Launch kernel using y-axis threading."));

static cl::opt<unsigned> MinWarpsPerExecUnit(
    "hipabi-min-warps-per-exec-unit", cl::init(1), cl::NotHidden,
    cl::desc("The minimum number of warps per execution unit"));

constexpr StringRef HIPABI_PREFIX = "kithip";
constexpr StringRef HIPABI_KERNEL_NAME_PREFIX = "kithip_loop_";

// LLVM variable name for the temporary embedded fat binary image. This will
// eventually be replaced with a global variable whose initializer is the actual
// device code.
constexpr StringRef HIPABI_DUMMY_FATBIN_NAME = "_hipabi.dummy_fatbin";

namespace {

/// @brief Is the given function an AMD GPU kernel.
/// @param F -- the Function to inspect.
/// @return true if the function is a kernel, false otherwise.
bool isAMDKernelFunction(const Function &Fn) {
  return Fn.getCallingConv() == CallingConv::AMDGPU_KERNEL;
}

std::string buildTargetFeatureString(StringRef ArchIdStr) {
  std::string FeaturesStr;

  using namespace AMDGPU;
  switch (parseArchAMDGCN(ArchIdStr)) {
  case GK_GFX1103:
    FeaturesStr +=
        "+16-bit-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot10-insts,"
        "+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+"
        "gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,";
    break;
  case GK_GFX90A:
    FeaturesStr += "+gfx90a-insts,+atomic-buffer-global-pk-add-f16-insts,"
                   "+atomic-fadd-rtn-insts,";
    [[fallthrough]];
  case GK_GFX908:
    FeaturesStr += "+dot3-insts,+dot4-insts,+dot5-insts,"
                   "+dot6-insts,+mai-insts,";
    [[fallthrough]];
  case GK_GFX906:
    FeaturesStr += "+dl-insts,+dot1-insts,+dot2-insts,+dot7-insts,";
    [[fallthrough]];
  case GK_GFX90C:
  case GK_GFX909:
  case GK_GFX904:
  case GK_GFX902:
  case GK_GFX900:
    FeaturesStr += "+gfx9-insts,";
    [[fallthrough]];
  case GK_GFX810:
  case GK_GFX805:
  case GK_GFX803:
  case GK_GFX802:
  case GK_GFX801:
    FeaturesStr += "+gfx8-insts,+16-bit-insts,+dpp,+s-memrealtime,";
    [[fallthrough]];
  case GK_GFX705:
  case GK_GFX704:
  case GK_GFX703:
  case GK_GFX702:
  case GK_GFX701:
  case GK_GFX700:
    FeaturesStr += "+ci-insts,";
    [[fallthrough]];
  case GK_GFX602:
  case GK_GFX601:
  case GK_GFX600:
    FeaturesStr += "+s-memtime-inst,";
    break;
  case GK_NONE:
    break;
  default:
    llvm_unreachable("Unhandled GPU!");
  }

  // TODO: feature is arch specific.  need to cross check.
  if (EnableXnack)
    FeaturesStr += "+xnack,";

  // TODO: feature is arch specific. need to cross-check.
  if (EnableSRAMECC)
    FeaturesStr += "+sramecc,";
  else
    FeaturesStr += "-sramecc,";

  // TODO: feature is arch specific. Meed to cross check.
  if (Use64ElementWavefront)
    FeaturesStr += "+wavefrontsize64";
  else
    FeaturesStr += "+wavefrontsize32";

  return FeaturesStr;
}

/// @brief Make calls within a function match the function's calling conv.
/// @param F -- The function to walk looking for calls.
/// @return void (calls within F will be modified)
void transformCallingConv(Function &F) {
  for (inst_iterator I = inst_begin(&F); I != inst_end(&F); I++) {
    if (auto CI = dyn_cast<CallInst>(&*I)) {
      Function *CF = CI->getCalledFunction();
      if (CI->getCallingConv() != CF->getCallingConv()) {
        LLVM_DEBUG(dbgs() << "\t\t\t-* updated calling convention to "
                          << "match '" << CF->getName() << "()'.\n");
        CI->setCallingConv(CF->getCallingConv());
      }
    }
  }
}

} // namespace

HipABIOptions::HipABIOptions() : GPUABIOptionsBase(TTO_Hip) {
  setArch(KITSUNE_HIP_ARCH_DEFAULT);

  // FIXME: This is here purely for debugging. Once we sort out how to best deal
  // with the threads per block value, this should go away.
  if (std::optional<std::string> ThreadsPBVar =
          sys::Process::GetEnv("KITHIP_THREADS_PER_BLOCK")) {
    if (getFixedThreadsPerBlock())
      errs() << "kitsune[hipabi]: Note that KITHIP_THREADS_PER_BLOCK is "
             << "overriding command line args.\n";
    setFixedThreadsPerBlock(std::stoi(ThreadsPBVar.value()));
  }
}

HipABIOptions *HipABIOptions::clone() const { return new HipABIOptions(*this); }

void HipABIOptions::readClOptions() {
  GPUABIOptionsBase::readClOptions();
  setArch(clGPUArch);
}

void HipABI::transformConstants(Function &Fn) {
  std::map<GetElementPtrInst *, GetElementPtrInst *> GEPMap;

  for (BasicBlock &BB : Fn) {
    for (Instruction &I : BB) {
      if (auto GEP = dyn_cast<GetElementPtrInst>(&I)) {
        if (auto PTy = dyn_cast<PointerType>(GEP->getType())) {
          unsigned AddrSpace = GEP->getAddressSpace();
          unsigned PtrAddrSpace = PTy->getAddressSpace();
          if (AddrSpace != PtrAddrSpace) {
            std::vector<Value *> opt_vec;
            for (Use &idx : GEP->indices())
              opt_vec.push_back(idx.get());
            ArrayRef<Value *> IdxList(opt_vec);
            Type *DestTy = GetElementPtrInst::getIndexedType(
                GEP->getSourceElementType(), IdxList);
            assert(DestTy && "GEP indices invalid!");
            GetElementPtrInst *NewGEP = GetElementPtrInst::Create(
                GEP->getSourceElementType(), GEP->getPointerOperand(), IdxList,
                GEP->getName() + ".asp", GEP);
            GEPMap[GEP] = NewGEP;
          }
        }
      }
    }
  }

  for (auto &iGEP : GEPMap) {
    GetElementPtrInst *OldGEP = iGEP.first;
    GetElementPtrInst *NewGEP = iGEP.second;
    std::vector<Use *> uses;
    for (Use &U : OldGEP->uses())
      uses.push_back(&U);
    for (Use *U : uses) {
      User *User = U->getUser();
      if (auto LI = dyn_cast<LoadInst>(User)) {
        LLVM_DEBUG(dbgs() << "\t\tpatching load instruction: " << *LI << "\n");
        LI->setOperand(LI->getPointerOperandIndex(), NewGEP);
        LLVM_DEBUG(dbgs() << "\t\t\t\tnew load: " << *LI << "\n");
      } else if (auto SI = dyn_cast<StoreInst>(User)) {
        LLVM_DEBUG(dbgs() << "\t\tpatching store instruction: " << *SI << "\n");
        SI->setOperand(SI->getPointerOperandIndex(), NewGEP);
        LLVM_DEBUG(dbgs() << "\t\t\t\tnew store: " << *SI << "\n");
      } else if (auto *Call = dyn_cast<CallBase>(User)) {
        unsigned argNo = Call->getArgOperandNo(U);
        LLVM_DEBUG(dbgs() << "\t\tpatching callable instruction: " << *Call
                          << "\n");
        // FIXME: This is not correct! The function operand should be
        // checked to see what address space it expects.
        Instruction *asCast =
            new AddrSpaceCastInst(NewGEP, OldGEP->getType(), "", Call);
        Call->setArgOperand(argNo, asCast);
        LLVM_DEBUG(dbgs() << "\t\t\t\tnew call: " << *Call << "\n");
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(User)) {
        LLVM_DEBUG(dbgs() << "\t\tpatching gep instruction:\n\t\t\t" << *GEP
                          << "\n");
        Instruction *asCast =
            new AddrSpaceCastInst(NewGEP, OldGEP->getType(), "", GEP);
        GEP->setOperand(GEP->getPointerOperandIndex(), asCast);
        LLVM_DEBUG(dbgs() << "\t\t\t\tnew gep:\n\t\t\t\t  " << *GEP << "\n");
      } else if (auto *PToI = dyn_cast<PtrToIntInst>(User)) {
        LLVM_DEBUG(dbgs() << "\t\tpatching ptrtoint instruction:\n\t\t\t"
                          << *PToI << "\n");
        Instruction *asCast =
            new AddrSpaceCastInst(NewGEP, OldGEP->getType(), "", PToI);
        PToI->setOperand(0, asCast);
        LLVM_DEBUG(dbgs() << "\t\t\t\tnew ptrtoint:\n\t\t\t\t  " << *PToI
                          << "\n");
      } else {
        LLVM_DEBUG(dbgs() << "Unexpected use: " << *U->get() << "\n");
        LLVM_DEBUG(dbgs() << "Unexpected user: " << *User << "\n");
        assert(false && "unexpected use/user of gep.");
      }
    }
    OldGEP->eraseFromParent();
  }
}

void HipABI::transformArguments(Function &Fn) {
  LLVMContext &Ctxt = Fn.getContext();
  std::vector<Type *> FnArgTypes(Fn.arg_size());
  for (auto &A : Fn.args()) {
    FnArgTypes[A.getArgNo()] = A.getType();
    if (isa<PointerType>(A.getType())) {
      LLVM_DEBUG(dbgs() << "\t\ttransforming argument: " << A << "\n");
      PointerType *NewPtrTy = PointerType::get(Ctxt, AMDGPUAS::GLOBAL_ADDRESS);
      // TODO: Better path here than mutate?
      A.mutateType(NewPtrTy);
      FnArgTypes[A.getArgNo()] = NewPtrTy;
      LLVM_DEBUG(dbgs() << "\t\t\tto: " << A << "\n");
    }
  }

  FunctionType *NewFTy = FunctionType::get(Fn.getReturnType(),
                                           ArrayRef<Type *>(FnArgTypes), false);
  Fn.mutateType(NewFTy->getPointerTo());
  // TODO: Need a better path here than mutate... We added this call to LLVM
  // to serve our testing and prototyping purposes.  Not sure there is a clean
  // (and easy to implement) way to accompish the same functionality...
  Fn.mutateValueType(NewFTy);
}

// --- Loop Outliner

/// @brief Return the work item ID for the calling thread. (thread index)
/// @param Builder - IR builder for code gen assistance.
/// @param ItemIndex - which work item dimension (x=0,y=1,z=2)
/// @param Low - Low-end of value range if known.
/// @param High -- High-end of value range if known.
Value *HipLoop::emitWorkItemId(IRBuilder<> &Builder, int ItemIndex) {
  switch (ItemIndex) {
  case 0:
    return Builder.CreateCall(KitHipWorkItemIdXFn, {}, "kern.witem.x");
    break;
  case 1:
    return Builder.CreateCall(KitHipWorkItemIdYFn, {}, "kern.witem.y");
    break;
  case 2:
    return Builder.CreateCall(KitHipWorkItemIdZFn, {}, "kern.witem.z");
    break;
  default:
    llvm_unreachable("unexpected item index!");
    return nullptr;
  }
}

/// @brief Return the work group ID for the calling thread. (block index)
/// @param Builder - IR builder for code gen assistance.
/// @param ItemIndex - which work item dimension (x=0,y=1,z=2)
Value *HipLoop::emitWorkGroupId(IRBuilder<> &Builder, int ItemIndex) {
  switch (ItemIndex) {
  case 0:
    return Builder.CreateCall(KitHipWorkGroupIdXFn, {}, "kern.wgroup.x");
    break;
  case 1:
    return Builder.CreateCall(KitHipWorkGroupIdYFn, {}, "kern.wgroup.y");
    break;
  case 2:
    return Builder.CreateCall(KitHipWorkGroupIdZFn, {}, "kern.wgroup.z");
    break;
  default:
    llvm_unreachable("unexpected item index!");
    return nullptr;
  }
}

/// @brief Return the work group size for the calling thread. (block size)
/// @param Builder - IR builder for code gen assistance.
/// @param ItemIndex - which work item dimension (x=0,y=1,z=2)
Value *HipLoop::emitWorkGroupSize(IRBuilder<> &Builder, int ItemIndex) {
  LLVMContext &Ctx = KernelModule.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Constant *IndexVal = ConstantInt::get(Int32Ty, ItemIndex);
  std::string WGName = "blockDim.";
  switch (ItemIndex) {
  case 0:
    WGName.append("x");
    break;
  case 1:
    WGName.append("y");
    break;
  case 2:
    WGName.append("z");
    break;
  default:
    llvm_unreachable("unexpected item index!");
  }
  Instruction *WorkGroupSizeCall =
      Builder.CreateCall(KitHipBlockDimFn, {IndexVal}, WGName);
  return WorkGroupSizeCall;
}

unsigned HipLoop::NextKernelID = 0;

HipLoop::HipLoop(Module &M, Module &KModule, const std::string &Name,
                 HipABI *LoopTarget)
    : LoopOutlineProcessor(M, KModule), TT(LoopTarget), KernelName(Name),
      KernelModule(KModule) {
  std::string UN = KernelName + "." + Twine(NextKernelID).str();
  NextKernelID++;
  KernelName = UN;

  LLVM_DEBUG(dbgs() << "hipabi: hip loop outliner creation:\n"
                    << "\ttransforming loop to kernel: " << KernelName
                    << "(...)\n"
                    << "\tdevice-side module name    : "
                    << KernelModule.getName() << "\n\n");

  LLVMContext &Ctx = KernelModule.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);

  NamedMDNode *IdentMetadata =
      KernelModule.getOrInsertNamedMetadata("llvm.ident");
  unsigned Major, Minor, Patch;
  LLVMGetVersion(&Major, &Minor, &Patch);
  std::string VersionStr = "kitsune/tapir/llvm " + std::to_string(Major) + "." +
                           std::to_string(Minor) + "." + std::to_string(Patch);
  Metadata *KitTapirIdentNode[] = {MDString::get(Ctx, VersionStr)};
  IdentMetadata->addOperand(MDNode::get(Ctx, KitTapirIdentNode));

  // We use ROCm/HSA/HIP entry points for various runtime calls.  These calls
  // are often at a lower level vs. user-facing entry points.  This follows
  // lower-level code generation details for HIP (that also include details
  // tucked into the HIP-centric header files as well a Clang lowering).

  // Get the local workitem ID for the calling thread.
  KitHipWorkItemIdFn = KernelModule.getOrInsertFunction(
      "__ockl_get_local_id",
      Int64Ty,  // return local thread id.
      Int32Ty); // axis/index select (x=0, y=1, z=2).

  // Get the work group ID for the calling thread.
  KitHipWorkGroupIdFn = KernelModule.getOrInsertFunction(
      "__ockl_get_group_id",
      Int64Ty,  // return local thread id.
      Int32Ty); // axis/index select (x=0, y=1, z=2).

  // Get the block size for the calling thread.
  KitHipBlockDimFn = KernelModule.getOrInsertFunction(
      "__ockl_get_local_size",
      Int64Ty,  // return local thread id.
      Int32Ty); // axis/index select (x=0, y=1, z=2).

  KitHipWorkItemIdXFn = /* threadIdx.x */
      Intrinsic::getDeclaration(&KernelModule, Intrinsic::amdgcn_workitem_id_x);
  KitHipWorkItemIdYFn = /* threadIdx.y */
      Intrinsic::getDeclaration(&KernelModule, Intrinsic::amdgcn_workitem_id_y);
  KitHipWorkItemIdZFn = /* threadIdx. z */
      Intrinsic::getDeclaration(&KernelModule, Intrinsic::amdgcn_workitem_id_z);

  KitHipWorkGroupIdXFn = /* blockIdx.x */
      Intrinsic::getDeclaration(&KernelModule,
                                Intrinsic::amdgcn_workgroup_id_x);
  KitHipWorkGroupIdYFn = /* blockIdx.y */
      Intrinsic::getDeclaration(&KernelModule,
                                Intrinsic::amdgcn_workgroup_id_y);
  KitHipWorkGroupIdZFn = /* blockIdx.z */
      Intrinsic::getDeclaration(&KernelModule,
                                Intrinsic::amdgcn_workgroup_id_z);

  // Get entry points into the Hip-centric portion of the Kitsune runtime.
  // NOTE: This needs to be sync'ed up with the tapir gpu utils on the
  // compiler side and the kitsune runtime as well (we should probably
  // share a header here for this but waiting for the configuration and
  // build system dust to settle...)
  KernelInstMixTy = StructType::get(Int64Ty,  // number of memory ops.
                                    Int64Ty,  // number of floating point ops.
                                    Int64Ty,  // number of integer ops.
                                    Int64Ty); // number of other ops.

  KitHipLaunchFn = M.getOrInsertFunction(
      "__kithip_launch_kernel",
      VoidPtrTy,                       // return an opaque stream
      VoidPtrTy,                       // fat-binary
      VoidPtrTy,                       // kernel name
      VoidPtrTy,                       // arguments
      Int64Ty,                         // trip count
      Int32Ty,                         // threads-per-block
      KernelInstMixTy->getPointerTo(), // instruction mix info
      VoidPtrTy);                      // opaque hip stream

  KitHipMemPrefetchFn =
      M.getOrInsertFunction("__kithip_mem_gpu_prefetch",
                            VoidPtrTy,  // return an opaque stream
                            VoidPtrTy,  // pointer to prefetch
                            VoidPtrTy); // use opaque stream.
}

HipLoop::~HipLoop() { /* no-op */ }

// TODO: Can we also transform the arguments into a different address space here
// and avoid our use of 'mutate' elsewhere in the code?
void HipLoop::setupLoopOutlineArgs(Function &F, ValueSet &HelperArgs,
                                   SmallVectorImpl<Value *> &HelperInputs,
                                   ValueSet &InputSet,
                                   const SmallVectorImpl<Value *> &LCArgs,
                                   const SmallVectorImpl<Value *> &LCInputs,
                                   const ValueSet &TLInputsFixed) {
  LLVM_DEBUG(dbgs() << "\n\n"
                    << "hipabi: SETTING UP LOOP OUTLINE ARGUMENTS FOR '"
                    << F.getName() << "()'.\n");

  // Add the loop control inputs -- the first parameter defines the extent of
  // the index space.
  {
    Argument *EndArg = cast<Argument>(LCArgs[1]);
    EndArg->setName(".kern.input_size"); // nice for debugging...
    HelperArgs.insert(EndArg);

    Value *InputVal = LCInputs[1];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // The second parameter defines the start of the index space.
  {
    Argument *StartArg = cast<Argument>(LCArgs[0]);
    StartArg->setName(".kern.start_idx");
    HelperArgs.insert(StartArg);

    Value *InputVal = LCInputs[0];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // The third parameter defines the grain size, if it is not constant.
  if (!isa<ConstantInt>(LCInputs[2])) {
    Argument *GrainsizeArg = cast<Argument>(LCArgs[2]);
    GrainsizeArg->setName(".kern.grain_size");
    HelperArgs.insert(GrainsizeArg);

    Value *InputVal = LCInputs[2];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // Add the loop-centric kernel parameters (i.e., variables/arrays
  // used in the loop body).
  for (Value *V : TLInputsFixed) {
    HelperArgs.insert(V);
    HelperInputs.push_back(V);
  }

  for (Value *V : HelperInputs) {
    OrderedInputs.push_back(V);
  }
}

unsigned HipLoop::getIVArgIndex(const Function &F, const ValueSet &Args) const {
  // The argument for the primary induction variable is the second input.
  return 1;
}

unsigned HipLoop::getLimitArgIndex(const Function &F,
                                   const ValueSet &Args) const {
  // The argument for the loop limit is the first input.
  return 0;
}

/// @brief Look for the given function in the device-side modules.
/// @param Fn - the function to resolve.
/// @param DevMod - Module containing the device-side routines (e.g. math).
/// @param KernelModule - Module containing the transformed device-side code.
/// @return The resolved function -- nullptr if not unresolved.
Function *HipLoop::resolveDeviceFunction(Function *Fn, bool enableFast) {
  if (Fn->isTargetIntrinsic()) {
    LLVM_DEBUG(dbgs() << "hipabi: function '" << Fn->getName()
                      << "()' resolved as a target-specific intrinsic.\n");
    return Fn;
  }

  // ROCm drags opencl back onto the stage...
  std::string OCLPrefix = "__ocml_";
  if (OCLPrefix == Fn->getName().str().substr(0, OCLPrefix.size() - 1)) {
    LLVM_DEBUG(dbgs() << "hipabi: skipping already prefixed function '"
                      << Fn->getName() << "())'.\n");
    return Fn;
  }

  std::string FnName;
  if (enableFast) {
    // TODO: need to support this.
    llvm_unreachable("hiabi: fast math call transformations not supported.\n");
  } else {
    FnName = OCLPrefix + StringSwitch<std::string>(Fn->getName().str())
                             .Case("acos", "acos_f64")
                             .Case("acosf", "acos_f32")
                             .Case("acosh", "acosh_f64")
                             .Case("acoshf", "acosh_f32")
                             .Case("asin", "asin_f64")
                             .Case("asinf", "asin_f32")
                             .Case("asinh", "asinh_f64")
                             .Case("asinhf", "asinh_f32")
                             .Case("atan2", "atan2_f64")
                             .Case("atan2f", "atan2_f32")
                             .Case("atan", "atan_f64")
                             .Case("atanf", "atahn_f32")
                             .Case("atanh", "atanh_f64")
                             .Case("atanhf", "atanh_f32")
                             .Case("cbrt", "cbrt_f64")
                             .Case("cbrtf", "cbrt_f32")
                             .Case("cos", "cos_f64")
                             .Case("cosf", "cos_f32")
                             .Case("cosh", "cosh_f64")
                             .Case("coshf", "cosh_f32")
                             .Case("erfc", "erfc_f64")
                             .Case("erfcf", "erfc_f32")
                             .Case("erf", "erf_f64")
                             .Case("erff", "erf_f32")
                             .Case("exp2", "exp2_f64")
                             .Case("exp2f", "exp2_f32")
                             .Case("exp", "exp_f64")
                             .Case("expf", "exp_f32")
                             .Case("expm1", "expm1_f64")
                             .Case("expm1f", "expm1_f32")
                             .Case("fmodf", "fmod_f32")
                             .Case("fmod", "fmod_f64")
                             .Case("hypotf", "hypot_f32")
                             .Case("hypot", "hypot_f64")
                             .Case("lgammaf", "lgamma_f32")
                             .Case("lgamma", "lgamma_f64")
                             .Case("llvm.cos.f32", "cos_f32")
                             .Case("llvm.cos.f64", "cos_f64")
                             .Case("llvm.exp.f32", "exp_f32")
                             .Case("llvm.exp.f64", "exp_f64")
                             .Case("llvm.fabs.f32", "fabs_f32")
                             .Case("llvm.fabs.f64", "fabs_f64")
                             .Case("llvm.fmod.f32", "fmod_f32")
                             .Case("llvm.fmod.f64", "fmod_f64")
                             .Case("llvm.maxnum.f32", "fmax_f32") // correct?
                             .Case("llvm.maxnum.f64", "fmax_f64") // correct?
                             .Case("llvm.minnum.f32", "fmin_f32") // correct?
                             .Case("llvm.minnum.f64", "fmin_f64") // correct?
                             .Case("llvm.pow.f32", "pow_f32")
                             .Case("llvm.pow.f64", "pow_f64")
                             .Case("llvm.sincos.f32", "sincos_f32")
                             .Case("llvm.sincos.f64", "sincos_f64")
                             .Case("llvm.sin.f32", "sin_f32")
                             .Case("llvm.sin.f64", "sin_f64")
                             .Case("llvm.sqrt.f32", "sqrt_f32")
                             .Case("llvm.sqrt.f64", "sqrt_f64")
                             .Case("llvm.tan.f32", "tan_f32")
                             .Case("llvm.tan.f64", "tan_f64")
                             .Case("llvm.tanh.f32", "tanh_f32 ")
                             .Case("llvm.tanh.f64", "tanh_f64")
                             .Case("log10f", "log10_f32")
                             .Case("log10", "log10_f64")
                             .Case("log1pf", "log1p_f32")
                             .Case("log1p", "log1p_f64")
                             .Case("log2f", "log2_f32")
                             .Case("log2", "log2_f64")
                             .Case("logf", "log_f32")
                             .Case("log", "log_f64")
                             .Case("powf", "pow_f32")
                             .Case("pow", "pow_f64")
                             .Case("sincosf", "sincos_f32")
                             .Case("sincos", "sincos_f64")
                             .Case("sinf", "sin_f32")
                             .Case("sinhf", "sinh_f32")
                             .Case("sinh", "sinh_f64")
                             .Case("sin", "sin_f64")
                             .Case("sqrtf", "sqrt_f32")
                             .Case("sqrt", "sqrt_f64")
                             .Case("tanf", "tan_f32")
                             .Case("tanhf", "tanh_f32")
                             .Case("tanh", "tanh_f64")
                             .Case("tan", "tan_f64")
                             .Case("tgammaf", "tgamma_f32")
                             .Case("tgamma", "tgamma_f64")
                             .Default("");
  }

  if (FnName == OCLPrefix) {
    if (Fn->isIntrinsic())
      return Fn;
    else
      return nullptr;
  }

  std::unique_ptr<Module> &DevMod = TT->getLibDeviceModule();
  if (Function *DevFn = DevMod->getFunction(FnName)) {
    LLVM_DEBUG(dbgs() << "\t\t\tresolved mapped function '" << FnName
                      << "' in device library module.\n");
    if (Function *KF = KernelModule.getFunction(FnName))
      return KF;

    Function *DeviceF =
        Function::Create(DevFn->getFunctionType(), DevFn->getLinkage(),
                         DevFn->getName(), KernelModule);
    DeviceF->setAttributes(DevFn->getAttributes());
    return DeviceF;
  } else {
    LLVM_DEBUG(dbgs() << "\t\t\t *unresolved* function '" << FnName
                      << "()'.  Not in device library???\n");
    return nullptr;
  }
}

/// @brief Transform the given function so it is ready for the final AMDGPU code
/// generation steps.
/// @param F - the function to transform.
/// @return
void HipLoop::transformForGCN(Function &F) {
  LLVM_DEBUG(dbgs() << "- transform '" << F.getName() << "()' "
                    << "for AMDGPU code generation.\n");

  std::map<CallInst *, CallInst *> Replaced;
  std::list<Function *> CalledFns;
  std::map<AllocaInst *, AddrSpaceCastInst *> AllocaReplaced;
  for (inst_iterator I = inst_begin(F); I != inst_end(F); I++) {
    if (auto CI = dyn_cast<CallInst>(&*I)) {
      Function *CF = CI->getCalledFunction();
      if (CF->isDeclaration()) {
        if (Function *DF = resolveDeviceFunction(CF, false /* no fast */)) {
          if (DF != CF) {
            CallInst *NCI = dyn_cast<CallInst>(CI->clone());
            NCI->setCalledFunction(DF);
            Replaced[CI] = NCI;
          }
        }
      } else {
        if (CF != &F) // no sneaky recursion please...
          CalledFns.push_back(CF);
      }
    } else if (auto AI = dyn_cast<AllocaInst>(&*I)) {
      const DataLayout &DL = KernelModule.getDataLayout();
      unsigned AllocaAS = DL.getAllocaAddrSpace();
      if (AI->getAddressSpace() != AllocaAS) {
        LLVM_DEBUG(dbgs() << "\t\t\ttransforming alloca address space from "
                          << AI->getAddressSpace() << " to " << AllocaAS
                          << ".\n");
        AllocaInst *NewAI =
            new AllocaInst(AI->getType(), AllocaAS, AI->getArraySize(),
                           AI->getAlign(), AI->getName());
        NewAI->insertBefore(AI);
        AddrSpaceCastInst *CastAI = new AddrSpaceCastInst(NewAI, AI->getType());
        AllocaReplaced[AI] = CastAI;
      }
    }
  }

  for (auto I : Replaced) {
    CallInst *CI = I.first;
    CallInst *NCI = I.second;
    NCI->insertAfter(CI);
    CI->replaceAllUsesWith(NCI);
    CI->eraseFromParent();
  }

  for (Function *Fn : CalledFns)
    transformForGCN(*Fn);

  for (auto I : AllocaReplaced) {
    AllocaInst *AI = I.first;
    AddrSpaceCastInst *AC = I.second;
    AC->insertAfter(AI);
    AI->replaceAllUsesWith(AC);
    AI->eraseFromParent();
  }

  LLVM_DEBUG(saveFunctionToFile(&F, F.getName().str(), ".hipabi.ll"));
}

void HipLoop::preProcessTapirLoop(TapirLoopInfo &TL, ValueToValueMapTy &VMap) {
  bool VerboseMode = TT->getOptions().getVerbose();
  if (VerboseMode) {
    errs() << "kitsune[hipabi]: pre-processing tapir loop.\n";
    errs() << "  - collecting global values from loop...\n";
  }

  // TODO: process loop prior to outlining to do GPU/HIP-specific things
  // like capturing global variables, etc.

  // Collect the top-level entities (Function, GlobalVariable, GlobalAlias
  // and GlobalIFunc) that are used in the outlined loop. Since the outlined
  // loop will live in the KernelModule, any GlobalValues will need to be
  // cloned into the KernelModule (with different details for the specific
  // type of value).
  std::set<GlobalValue *> UsedGlobalValues;
  Loop &L = *TL.getLoop();
  for (Loop *SL : L)
    for (BasicBlock *BB : SL->blocks())
      tapir::collectGlobalValues(*BB, UsedGlobalValues);
  for (BasicBlock *BB : L.blocks())
    tapir::collectGlobalValues(*BB, UsedGlobalValues);

  if (VerboseMode) {
    errs() << "  - global address space (amdgpu): " << AMDGPUAS::GLOBAL_ADDRESS
           << "\n";
    if (UsedGlobalValues.size() > 0)
      errs() << "  - cloning collected globals into kernel module.\n";
    else
      errs() << "  - no globals collected by loop analysis.\n";
  }

  // TODO: Need to work on making sure we understand the nuances here for
  // address space selection. In some cases, wrong address spaces seem to cause
  // crashes, in others they are performance optimizations, and sometimes they
  // almost seem to be no-ops... Some of the AMD documentation details seem
  // incomplete.
  //
  //   See: https://llvm.org/docs/AMDGPUUsage.html#amdgpu-address-spaces-table.
  //
  // Clone global variables (TODO: and aliases).
  //
  for (GlobalValue *V : UsedGlobalValues) {
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
      StringRef GVName = GV->getName();
      Type *GVType = GV->getValueType();
      GlobalVariable *NewGV = nullptr;
      if (GV->isConstant()) {
        if (VerboseMode)
          errs() << "    - constant: " << GVName << "\n";
        // If the global variable is a constant we can clone it into the device
        // module along with its initializer where it will be treated as an
        // internal variable. There is no coordination with the host.
        // TODO: make sure this is sound!
        NewGV = new GlobalVariable(KernelModule, GVType, true /*isConstant*/,
                                   GlobalValue::InternalLinkage,
                                   GV->getInitializer(), GVName + "_devvar",
                                   nullptr, GlobalValue::NotThreadLocal,
                                   AMDGPUAS::CONSTANT_ADDRESS);
        NewGV->setDSOLocal(GV->isDSOLocal());
      } else {
        if (VerboseMode)
          errs() << "    - non-constant: " << GVName << "\n";
        // If the global is not constant, we will need to create a device-side
        // version that will have the host-side value copied over prior to
        // launching the kernel.
        NewGV = new GlobalVariable(
            KernelModule, GVType, false /*isConstant*/,
            GlobalValue::LinkageTypes::ExternalLinkage,
            Constant::getNullValue(GVType), GVName + "_devvar",
            /* insertBefore */ nullptr, GlobalValue::NotThreadLocal,
            AMDGPUAS::GLOBAL_ADDRESS,
            /* externally initialized */ true);

        // HIP (appears) to require protected visibility! Without this the
        // runtime won't be able to find the global variable for host <-> device
        // transfers.
        NewGV->setVisibility(GlobalValue::ProtectedVisibility);

        // It is not clear what is tripping up the dso_local attribute, but it
        // seems to be required in this case.
        NewGV->setDSOLocal(true);

        Type *PtrTy = PointerType::getUnqual(M.getContext());
        GlobalVariable *DevVarPtr = new GlobalVariable(
            M, PtrTy, /*isConstant*/ false, GlobalValue::ExternalWeakLinkage,
            /*initializer*/ nullptr, GVName + ".devptr",
            /*insertBefore*/ nullptr, GlobalValue::NotThreadLocal,
            /*addrspace*/ 0, /*externallyInitialized*/ true);

        // Flag the GV for post-processing (e.g., insert copy calls).
        TT->pushGlobalVariable(GV);
      }

      NewGV->setAlignment(GV->getAlign());
      VMap[GV] = NewGV;
    } else if (isa<GlobalAlias>(V)) {
      llvm_unreachable("hipabi: GlobalAlias support not implemented!");
    }
  }

  // Create declarations for all functions first. These may be needed in the
  // global variables and aliases.
  for (GlobalValue *G : UsedGlobalValues) {
    if (Function *F = dyn_cast<Function>(G)) {
      Function *DeviceF = KernelModule.getFunction(F->getName());
      if (not DeviceF) {
        DeviceF = Function::Create(F->getFunctionType(),
                                   GlobalValue::LinkageTypes::ExternalLinkage,
                                   0, F->getName(), &KernelModule);
        if (VerboseMode) {
          errs() << "    - declare device function '" << F->getName()
                 << "()'\n";
        }
      }

      auto NewFArgIt = DeviceF->arg_begin();
      for (auto &Arg : F->args()) {
        StringRef ArgName = Arg.getName();
        NewFArgIt->setName(ArgName);
        VMap[&Arg] = &(*NewFArgIt++);
      }
      VMap[F] = DeviceF;
    }
  }

  // FIXME: Support GlobalIFunc at some point. This is a GNU extension, so we
  // may not want to support it at all, but just in case, this is here.
  for (GlobalValue *V : UsedGlobalValues)
    if (isa<GlobalIFunc>(V))
      llvm_unreachable("hipabi: GlobalIFunc not yet supported.");

  // Now clone any function bodies that need to be cloned. This should be
  // done as late as possible so that the VMap is populated with any other
  // global values that need to be remapped.
  LLVM_DEBUG(dbgs() << "\t*- cloning/creating device-side functions...\n");
  for (GlobalValue *v : UsedGlobalValues) {
    if (Function *F = dyn_cast<Function>(v)) {
      if (F->size() && not F->isIntrinsic()) {
        SmallVector<ReturnInst *, 8> Returns;
        Function *DeviceF = cast<Function>(VMap[F]);
        CloneFunctionInto(DeviceF, F, VMap,
                          CloneFunctionChangeType::DifferentModule, Returns);
        if (VerboseMode)
          errs() << "    - cloned '" << F->getName() << "()'.\n";

        DeviceF->removeFnAttr("target-cpu");
        DeviceF->removeFnAttr("target-features");
        DeviceF->removeFnAttr("tune-cpu");
        DeviceF->removeFnAttr(Attribute::UWTable);
        DeviceF->addFnAttr(Attribute::NoUnwind);
        DeviceF->addFnAttr(Attribute::AlwaysInline);
      }
    }
  }
}

void HipLoop::postProcessOutline(TapirLoopInfo &TLI, TaskOutlineInfo &Out,
                                 ValueToValueMapTy &VMap) {
  // addSyncToOutlineReturns(TLI, Out, VMap);
  Task *T = TLI.getTask();
  Loop *TL = TLI.getLoop();
  TapirLoopHints Hints(TL);
  // unsigned TPB = Hints.getThreadsPerBlock();

  BasicBlock *Entry = cast<BasicBlock>(VMap[TL->getLoopPreheader()]);
  BasicBlock *Header = cast<BasicBlock>(VMap[TL->getHeader()]);
  BasicBlock *Exit = cast<BasicBlock>(VMap[TLI.getExitBlock()]);
  PHINode *PrimaryIV = cast<PHINode>(VMap[TLI.getPrimaryInduction().first]);
  Value *PrimaryIVInput = PrimaryIV->getIncomingValueForBlock(Entry);

  TT->pushSR(T->getDetach()->getSyncRegion());

  // We no longer need the cloned sync region.
  Instruction *ClonedSyncReg =
      cast<Instruction>(VMap[T->getDetach()->getSyncRegion()]);
  ClonedSyncReg->eraseFromParent();

  // Get the kernel function for this loop and clean up any stray (target
  // related) attributes. Because of the way we compile the code, those
  // attributes will only be relevant for the host.
  Function *KernelF = Out.Outline;
  KernelF->setName(KernelName);

  // Remove any attributes that are only relevant for the host.
  KernelF->removeFnAttr("target-cpu");
  KernelF->removeFnAttr("target-features");
  KernelF->removeFnAttr("tune-cpu");

  KernelF->removeFnAttr(Attribute::UWTable);

  // Specify the minimum and maximum flat work group sizes that will be used
  // when the kernel is dispatched.
  std::string AttrVal;
  AttrVal = std::string("128,1024");
  KernelF->addFnAttr("amdgpu-flat-work-group-size", AttrVal);

#if 0 // DISABLED FOR TESTING
  unsigned MaxThreadsPerBlock = TT->getOptions().getMaxThreadsPerBlock();
  unsigned DefaultThreadsPerBlock = TT->getOptions().getFixedThreadsPerBlock();

  // Check for programmer-provided launch attribute...
  if (TPB > 0 && TPB <= MaxThreadsPerBlock) {
    AttrVal = std::string("1,") + utostr(TPB);
    KernelF->addFnAttr("amdgpu-flat-work-group-size", AttrVal);

    if (UseYLaunch)
      AttrVal = std::string("1,") + utostr(TPB) + std::string(",1");
    else
      AttrVal = utostr(TPB) + std::string(",1,1");
  } else if (DefaultThreadsPerBlock > 0 &&
             DefaultThreadsPerBlock <= MaxThreadsPerBlock) {
    // Check for command line spec.
    AttrVal = std::string("1,") + utostr(DefaultThreadsPerBlock);
    KernelF->addFnAttr("amdgpu-flat-work-group-size", AttrVal);

    if (UseYLaunch)
      AttrVal = std::string("1,") + utostr(DefaultThreadsPerBlock)
                    + std::string(",1");
    else
      AttrVal = utostr(DefaultThreadsPerBlock) + std::string(",1,1");
  } else {
    // Use defaults...
    AttrVal = std::string("1,") + utostr(MaxThreadsPerBlock);
    KernelF->addFnAttr("amdgpu-flat-work-group-size", AttrVal);

    if (UseYLaunch)
      AttrVal = std::string("1,") + utostr(MaxThreadsPerBlock) +
                std::string(",1");
    else
      AttrVal = utostr(MaxThreadsPerBlock) + std::string(",1,1");
  }
  // Attribute falls through from above conditionals...
  KernelF->addFnAttr("amdgpu-max-num-workgroups", AttrVal);
#endif

  KernelF->addFnAttr(Attribute::NoUnwind);
  if (ROCmABITarget == ROCm_ABI_V5)
    KernelF->addFnAttr("uniform-work-group-size", "true");

  // AMD requires that the kernel function have protected visiblity otherwise
  // AMD's runtime is unable to find the kernel function at runtime. This, in
  // turn requires the function to have external linkage. In case the function
  // gets here with a different linkage type, just override it.
  KernelF->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
  KernelF->setVisibility(GlobalValue::VisibilityTypes::ProtectedVisibility);
  KernelF->setCallingConv(CallingConv::AMDGPU_KERNEL);
  KernelF->addFnAttr("no-trapping-math", "true");
  KernelF->addFnAttr("target-cpu", TT->getOptions().getArch());
  KernelF->addFnAttr(Attribute::MustProgress);
  std::string targetFeaturesStr =
      buildTargetFeatureString(TT->getOptions().getArch());
  KernelF->addFnAttr("target-features", targetFeaturesStr);

  // Verify that the Thread ID corresponds to a valid iteration. Because
  // Tapir loops use canonical induction variables, valid iterations range
  // from 0 to the loop limit with stride 1. The End argument encodes the
  // loop limit. Get end and grain size arguments
  Argument *End;
  Value *Grainsize;
  {
    // TODO: We really only want a grain size of 1 for now...
    auto OutlineArgsIter = KernelF->arg_begin();
    // End argument is the first LC arg.
    End = &*OutlineArgsIter++;

    // Get the grain size value, which is either constant or the third LC
    // arg.
    // if (unsigned ConstGrainsize = TLI.getGrainsize())
    //  Grainsize = ConstantInt::get(PrimaryIV->getType(), ConstGrainsize);
    // else
    Grainsize =
        ConstantInt::get(PrimaryIV->getType(), DefaultGrainSize.getValue());
  }

  IRBuilder<> Builder(Entry->getTerminator());
  // Get the thread ID for this invocation of Helper.
  //
  // This is the classic thread ID calculation:
  //      i = blockDim.x * blockIdx.x + threadIdx.x;
  // For now we only generate 1-D thread IDs.
  Value *ThreadIdx;
  Value *BlockDim;

  if (not UseYLaunch) {
    ThreadIdx = Builder.CreateIntCast(
        emitWorkItemId(Builder, 0), PrimaryIV->getType(), false, "kern.tid.x");
    BlockDim =
        Builder.CreateIntCast(emitWorkGroupSize(Builder, 0),
                              PrimaryIV->getType(), false, "kern.blkdim.x");
  } else {
    ThreadIdx = Builder.CreateIntCast(
        emitWorkItemId(Builder, 1), PrimaryIV->getType(), false, "kern.tid.y");
    BlockDim =
        Builder.CreateIntCast(emitWorkGroupSize(Builder, 1),
                              PrimaryIV->getType(), false, "kern.blkdim.y");
  }

  Value *BlockIdx = Builder.CreateIntCast(
      emitWorkGroupId(Builder, 0), PrimaryIV->getType(), false, "kern.blkid.x");

  Value *ThreadID = Builder.CreateIntCast(
      Builder.CreateAdd(
          ThreadIdx,
          Builder.CreateMul(BlockIdx, BlockDim, ".kern.blk_offset.x"),
          ".kern.tid"),
      PrimaryIV->getType(), false, ".kern.thread_id");

  // NOTE/TODO: Assuming that the grainsize is fixed at 1 for the
  // current codegen...
  // ThreadID = Builder.CreateMul(ThreadID, Grainsize);
  Value *ThreadEnd = Builder.CreateAdd(ThreadID, Grainsize, ".kern.last_idx");
  Value *Cond = Builder.CreateICmpUGE(ThreadID, End, ".kern.at_end");
  ReplaceInstWithInst(Entry->getTerminator(),
                      BranchInst::Create(Exit, Header, Cond));

  // Replace the loop's induction variable with the GPU thread id.
  PrimaryIVInput->replaceAllUsesWith(ThreadID);

  // Update cloned loop condition to use the thread-end value.
  unsigned TripCountIdx = 0;
  ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TLI.getCondition()]);
  if (ClonedCond->getOperand(0) != End)
    ++TripCountIdx;
  assert(ClonedCond->getOperand(TripCountIdx) == End &&
         "End argument not used in condition!");
  ClonedCond->setOperand(TripCountIdx, ThreadEnd);
}

std::unique_ptr<Module> HipABI::loadBCFile(const std::string &BCFile) {
  LLVMContext &Ctx = KernelModule.getContext();
  SMDiagnostic SMD;
  LLVM_DEBUG(dbgs() << "\tloading bitcode file: " << BCFile << "...\n");
  std::unique_ptr<Module> BCM = parseIRFile(BCFile, SMD, Ctx);
  if (not BCM)
    report_fatal_error("Failed to parse bitcode file!");
  return BCM;
}

bool HipABI::linkInModule(std::unique_ptr<Module> &Mod) {
  assert(Mod != nullptr && "unexpected null module ptr!");

  // At this point we are ready to link in the device-side module for the final
  // steps of the target transformation. This basically completes resolution
  // for device-side calls that typically come from the GPU software stack
  // (e.g., the GPU math calls).
  Linker L(KernelModule);

  if (L.linkInModule(std::move(Mod), Linker::LinkOnlyNeeded))
    // TODO: Is there a way to get details here about why the link failed? For
    // now, just use a fatal error until more details can be provided.
    report_fatal_error("Failed to link in HipABI module!");
  else
    return true;
}

void HipLoop::remapData(ValueToValueMapTy &VMap) {
  for (auto &V : OrderedInputs)
    if (Value *MappedV = VMap[V])
      V = MappedV;
}

void HipLoop::processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                                      DominatorTree &DT) {
  LLVM_DEBUG(dbgs() << "hiploop: processing outlined loop call...\n"
                    << "\tkernel name: " << KernelName << "\n");

  LLVMContext &Ctx = M.getContext();

  // NOTE: If we are dealing with loop nests with multiple targets (in this case
  // only a CPU-target w/ a nested GPU target is supported) we can end up with
  // multiple calls to the outlined loop (which has been setup for dead code
  // elimination) but can cause invalid IR that trips us up when handling the
  // GPU module code generation. This is a challenge in the Tapir design that
  // was not geared to handle some of the nuances of GPU target transformations
  // (and code gen).  To address this, we need to do some clean up to keep the
  // IR correct (or the verifier will fail on us...).  Specifically, we can no
  // longer depend upon DCE as it runs too late in the GPU transformation
  // process...
  //
  // TODO: This code can be shared between the cuda and hip targets...
  //
  Function *TargetKF = KernelModule.getFunction(KernelName);
  std::list<Instruction *> RemoveList;
  if (TargetKF) {
    LLVM_DEBUG(dbgs() << "\t*- searching for 'dangling' outline calls...\n");
    for (Use &U : TargetKF->uses()) {
      if (auto *Inst = dyn_cast<Instruction>(U.getUser())) {
        LLVM_DEBUG(dbgs() << "\t\t- remove use: " << *Inst << "\n");
        if (Inst != TOI.ReplCall)
          RemoveList.push_back(Inst);
      }
    }
  }

  for (auto I : RemoveList)
    I->eraseFromParent();

  // Make a pass to prep for GCN code generation...
  LLVM_DEBUG(dbgs() << "\t*- transform kernel for GCN code gen.\n");
  Function &F = *KernelModule.getFunction(KernelName.c_str());
  transformForGCN(F);

  // Create two builders -- one inserts code into the entry block
  // (e.g., new "up-front" allocas) and the other is for generating
  // new code into a split BB.
  Function *Parent = TOI.ReplCall->getFunction();
  BasicBlock &EntryBB = Parent->getEntryBlock();
  IRBuilder<> EntryBuilder(&EntryBB.front());

  BasicBlock *RCBB = TOI.ReplCall->getParent();
  BasicBlock *NewBB = RCBB->splitBasicBlock(TOI.ReplCall);
  IRBuilder<> NewBuilder(&NewBB->front());

  // TODO: There is some potential here to share this code across both the hip
  // and cuda transforms.
  LLVM_DEBUG(dbgs() << "\t*- code gen packing of " << OrderedInputs.size()
                    << " kernel args.\n");
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  ArrayType *ArrayTy = ArrayType::get(VoidPtrTy, OrderedInputs.size());
  Value *ArgArray = EntryBuilder.CreateAlloca(ArrayTy);

  Value *NullPtr = ConstantPointerNull::get(PointerType::getUnqual(Ctx));
  Value *HipStream = ConstantPointerNull::get(PointerType::getUnqual(Ctx));

  unsigned int i = 0;
  for (Value *V : OrderedInputs) {
    Value *VP = EntryBuilder.CreateAlloca(V->getType());
    NewBuilder.CreateStore(V, VP);
    Value *VoidVPtr = NewBuilder.CreateBitCast(VP, VoidPtrTy);
    Value *ArgPtr =
        NewBuilder.CreateConstInBoundsGEP2_32(ArrayTy, ArgArray, 0, i);
    NewBuilder.CreateStore(VoidVPtr, ArgPtr);
    i++;

    if (CodeGenPrefetch && V->getType()->isPointerTy()) {
      LLVM_DEBUG(dbgs() << "\t\t- code gen prefetch for kernel arg #" << i - 1
                        << "\n");
      Value *VAS = NewBuilder.CreatePointerBitCastOrAddrSpaceCast(V, VoidPtrTy);
      HipStream = NewBuilder.CreateCall(KitHipMemPrefetchFn, {VAS, HipStream});
    }
  }

  // The next step is prep for the actual kernel launch call via
  // the kitsune runtime.  We have to add some extra levels of
  // pointers to match API details, deal with some potential
  // type mismatches, build a dummy pointer for the yet-to-be-created
  // fat binary, etc...
  const DataLayout &DL = M.getDataLayout();
  Value *argsPtr =
      NewBuilder.CreateConstInBoundsGEP2_32(ArrayTy, ArgArray, 0, 0);

  // Generate a call to launch the kernel.
  Constant *KNameCS = ConstantDataArray::getString(Ctx, KernelName);
  GlobalVariable *KNameGV =
      new GlobalVariable(M, KNameCS->getType(), true,
                         GlobalValue::PrivateLinkage, KNameCS, ".kern.name");
  KNameGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  Type *StrTy = KNameGV->getType();
  Constant *Zeros[] = {ConstantInt::get(DL.getIndexType(StrTy), 0),
                       ConstantInt::get(DL.getIndexType(StrTy), 0)};
  Constant *KNameParam =
      ConstantExpr::getGetElementPtr(KNameGV->getValueType(), KNameGV, Zeros);

  // We place *all* transformed tapir loops from the input module into a
  // single GPU target module.  At this point we can not create a complete
  // fat binary image.  However, we have all the important info for the
  // current loop so we use a 'dummy' (null) fat binary for code gen at
  // this point -- we'll post-process the module to clean this up after
  // we've processed all tapir loops.
  (void)tapir::getOrInsertFBGlobal(M, HIPABI_DUMMY_FATBIN_NAME, VoidPtrTy);

  // Deal with type mismatches for the trip count.  A difference
  // introduced via the input source details and the runtime's
  // API type signature for the lanuch.
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Value *TripCount = OrderedInputs[0];
  Value *CastTripCount;
  if (TripCount->getType() != Int64Ty)
    // It is not clear that this is actually signed.
    CastTripCount = NewBuilder.CreateIntCast(TripCount, Int64Ty, true);
  else
    CastTripCount = TripCount; // Simplify cases in launch code gen below...

  // At this point we need a threads-per-block value for the launch call. The
  // runtime will determine this value if ThreadsPerBlock is zero but it can
  // also be overridden via kitsune's forall launch attribute. The catch here is
  // the launch attribute's value for this is flexible and be a computed
  // expression vs. a compile-time constant. For this first step of creating the
  // kernel launch, we take the path of a runtime configuration vs. an
  // attributed launch. This will get patched up as needed when we post-process
  // the module and replace the DummyFBPtr (as we will also need to replace the
  // kernel launch call parameter for threads-per-block if an attributed
  // expression is present). See postProcessModule()'s stage of finalizing the
  // launch calls for details.
  TapirLoopHints Hints(TL.getLoop());
  unsigned TPBHint = Hints.getThreadsPerBlock();
  unsigned DefaultThreadsPerBlock = TT->getOptions().getFixedThreadsPerBlock();
  Value *TPBValue;

  if (TPBHint != 0)
    TPBValue = ConstantInt::get(Type::getInt32Ty(Ctx), TPBHint);
  else if (DefaultThreadsPerBlock != 0)
    TPBValue = ConstantInt::get(Type::getInt32Ty(Ctx), DefaultThreadsPerBlock);
  else
    TPBValue = ConstantInt::get(Type::getInt32Ty(Ctx), 0);

  LLVM_DEBUG(dbgs() << "\tgathering kernel instruction mix....\n");
  tapir::KernelInstMixData InstMix = tapir::getKernelInstructionMix(F);
  LLVM_DEBUG(
      dbgs() << "\tinstruction mix:\n"
             << "      memory ops      : " << InstMix.numMemoryOps << "\n"
             << "      flop count      : " << InstMix.numFlops << "\n"
             << "      integer op count: " << InstMix.numIntOps << "\n"
             << "      other ops count : " << InstMix.numOtherOps << "\n\n");

  Constant *InstructionMix = ConstantStruct::get(
      KernelInstMixTy, ConstantInt::get(Int64Ty, InstMix.numMemoryOps),
      ConstantInt::get(Int64Ty, InstMix.numFlops),
      ConstantInt::get(Int64Ty, InstMix.numIntOps),
      ConstantInt::get(Int64Ty, InstMix.numOtherOps));

  AllocaInst *AI = EntryBuilder.CreateAlloca(KernelInstMixTy);
  NewBuilder.CreateStore(InstructionMix, AI);

  LLVM_DEBUG(dbgs() << "\t*- code gen kernel launch...\n");
  HipStream = NewBuilder.CreateCall(
      KitHipLaunchFn,
      {NullPtr, KNameParam, argsPtr, CastTripCount, TPBValue, AI, HipStream});
  Type *VoidTy = Type::getVoidTy(Ctx);
  FunctionCallee KitHipSyncFn =
      M.getOrInsertFunction("__kithip_sync_thread_stream", VoidTy, VoidPtrTy);
  (void)NewBuilder.CreateCall(KitHipSyncFn, {HipStream});

  TOI.ReplCall->eraseFromParent();
  LLVM_DEBUG(dbgs() << "*** finished processing outlined call.\n");
}

// As is the pattern with the GPU targets, the HipABI is setup to process all
// Tapir constructs within a given input Module (M). It then creates a
// corresponding module that contains the transformed device-side code. This is
// the KernelModule that is created below in the target constructor.
HipABI::HipABI(Module &M, const HipABIOptions &opts)
    : TapirTarget(M, opts),
      KernelModule(
          tapir::concat(HIPABI_PREFIX + sys::path::filename(M.getName())),
          M.getContext()) {
  bool VerboseMode = opts.getVerbose();
  if (VerboseMode) {
    dbgs() << "'hip' tapir target options:\n";
    dbgs() << "  Runtime verbose:     " << opts.getRuntimeVerbose() << "\n";
    dbgs() << "  GPU arch:            " << opts.getArch() << "\n";
    dbgs() << "  Optimization level:  " << opts.getOptLevel() << "\n";
    dbgs() << "  FP Fusion:           " << opts.getFPOpFusionMode() << "\n";
    dbgs() << "  Fixed threads/block: " << opts.getFixedThreadsPerBlock()
           << "\n";
    dbgs() << "  Max threads/block:   " << opts.getMaxThreadsPerBlock() << "\n";
    dbgs() << "  Target features:     " << opts.getTargetFeatures() << "\n";
  }

  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  PointerType *CharPtrTy = PointerType::getUnqual(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  KitHipGetGlobalSymbolFn =
      M.getOrInsertFunction("__kithip_get_global_symbol",
                            VoidPtrTy,  // return the device pointer
                            VoidPtrTy,  // fat binary
                            CharPtrTy); // symbol name
  KitHipMemcpySymbolToDevFn =
      M.getOrInsertFunction("__kithip_memcpy_sym_to_device",
                            VoidTy,    // no return
                            VoidPtrTy, // host pointer
                            VoidPtrTy, // device pointer
                            Int64Ty);  // number of bytes to copy
  KitHipSyncFn = M.getOrInsertFunction("__kithip_sync_thread_stream", VoidTy,
                                       VoidPtrTy); // no return, nor parameters
  // Build the details we need for the AMDGPU/HIP target.
  std::string ArchString = "amdgcn";
  Triple TargetTriple(ArchString, "amd", "amdhsa");
  std::string Error;
  const Target *AMDGPUTarget =
      TargetRegistry::lookupTarget("", TargetTriple, Error);
  if (not AMDGPUTarget) {
    errs() << "hipabi: target lookup failed! '" << Error << "'\n";
    report_fatal_error("hipabi: unable to find registered HIP target. "
                       "Was LLVM built with the AMDGPU target enabled?");
  }

  // TODO: Something in the Tapir pipeline creates an excessive
  // number of "ABI" objects.
  static bool ShownOnce = false;
  if (VerboseMode && not ShownOnce) {
    errs() << "kitsune[hipabi]: create hip/amdgpu abi 'transformer'.\n";
    errs() << "  - target triple: " << TargetTriple.str() << "\n";
    errs() << "  - target gpu architecture: " << getOptions().getArch() << "\n";
    ShownOnce = true;
  }

  SmallString<255> NewModuleName(ArchString + KernelModule.getName().str());
  sys::path::replace_extension(NewModuleName, ".amdgcn");
  KernelModule.setSourceFileName(NewModuleName.c_str());

  CodeGenOptLevel TargetOptLevel;
  CodeModel::Model TargetCodeModel = CodeModel::Small; // ignored???

  switch (Level.getSpeedupLevel()) {
  case 0:
    TargetOptLevel = CodeGenOptLevel::None;
    break;
  case 1:
    TargetOptLevel = CodeGenOptLevel::Less;
    break;
  case 2:
    TargetOptLevel = CodeGenOptLevel::Default;
    break;
  case 3:
    TargetOptLevel = CodeGenOptLevel::Aggressive;
    break;
  default:
    llvm_unreachable("cuabi: unknown speed up level!");
    break;
  }

  std::string Features = buildTargetFeatureString(getOptions().getArch());
  TargetOptions Options;
  Options.UseInitArray = true;
  Options.EmitAddrsig = true;
  Options.AllowFPOpFusion = getOptions().getFPOpFusionMode();
  AMDTargetMachine = AMDGPUTarget->createTargetMachine(
      TargetTriple.getTriple(), getOptions().getArch(), Features, Options,
      Reloc::Static, TargetCodeModel, TargetOptLevel);

  LLVM_DEBUG(dbgs() << "\ttarget feature string:\n\t\t"
                    << AMDTargetMachine->getTargetFeatureString() << "\n\n");
  KernelModule.setTargetTriple(TargetTriple.str());
  KernelModule.setDataLayout(AMDTargetMachine->createDataLayout());
  ROCmModulesLoaded = false;
}

HipABI::~HipABI() { /* no-op */ }

const HipABIOptions &HipABI::getOptions() const {
  return cast<HipABIOptions>(opts);
}

std::unique_ptr<Module> &HipABI::getLibDeviceModule() {
  if (not LibDeviceModule) {
    LLVMContext &Ctx = KernelModule.getContext();
    SMDiagnostic SMD;

    // TODO: should we add flags to control some of these "on"/"off"
    // bitcode options exposed via command line args?
    std::initializer_list<std::string> BaseBCFiles = {
        "hip.bc",
        "ocml.bc",
        "ockl.bc",
        "oclc_daz_opt_off.bc",
        "oclc_unsafe_math_off.bc",
        "oclc_finite_only_off.bc",
        "oclc_correctly_rounded_sqrt_on.bc",
        //"opencl.bc", // printf lives here...
    };

    std::list<std::string> ROCmBCFiles;
    for (std::string BCFile : BaseBCFiles)
      ROCmBCFiles.push_back(BCFile);

    if (Use64ElementWavefront)
      ROCmBCFiles.push_back("oclc_wavefrontsize64_on.bc");
    else
      ROCmBCFiles.push_back("oclc_wavefrontsize64_off.bc");

    // Pick the corresponding bitcode file for the target architecture.
    // TODO: Add support for multiple architectures in a single transform.
    StringRef gpuArch = getOptions().getArch();
    if (gpuArch == "gfx900")
      ROCmBCFiles.push_back("oclc_isa_version_900.bc");
    else if (gpuArch == "gfx902")
      ROCmBCFiles.push_back("oclc_isa_version_902.bc");
    else if (gpuArch == "gfx904")
      ROCmBCFiles.push_back("oclc_isa_version_904.bc");
    else if (gpuArch == "gfx906")
      ROCmBCFiles.push_back("oclc_isa_version_906.bc");
    else if (gpuArch == "gfx908")
      ROCmBCFiles.push_back("oclc_isa_version_908.bc");
    else if (gpuArch == "gfx90a")
      ROCmBCFiles.push_back("oclc_isa_version_90a.bc");
    else if (gpuArch == "gfx90c")
      ROCmBCFiles.push_back("oclc_isa_version_90c.bc");
    else if (gpuArch == "gfx1103")
      ROCmBCFiles.push_back("oclc_isa_version_1103.bc");
    else {
      errs() << "unsupported amdgpu archicture target: " << gpuArch << ".\n";
      report_fatal_error("fatal error!");
    }

    if (ROCmABITarget == ROCm_ABI_V4)
      ROCmBCFiles.push_back("oclc_abi_version_400.bc");
    else if (ROCmABITarget == ROCm_ABI_V5)
      ROCmBCFiles.push_back("oclc_abi_version_500.bc");
    else
      llvm_unreachable("unhandled ROCm ABI version!");

    LLVM_DEBUG(dbgs() << "\tpre-loading AMDGCN device bitcode files.\n");
    for (std::string BCFile : ROCmBCFiles) {
      const std::string GCNFile = std::string(KITSUNE_HIP_BITCODE_DIR) + BCFile;
      LLVM_DEBUG(dbgs() << "\t\t* " << GCNFile << "\n");
      if (LibDeviceModule == nullptr) {
        LibDeviceModule = parseIRFile(GCNFile, SMD, Ctx);
        if (LibDeviceModule == nullptr) {
          SMD.print(GCNFile.c_str(), errs());
          report_fatal_error("Failed to parse bitcode file!");
        }
      } else {
        std::unique_ptr<Module> BCModule;
        BCModule = parseIRFile(GCNFile, SMD, Ctx);
        if (BCModule == nullptr) {
          SMD.print(GCNFile.c_str(), errs());
          report_fatal_error("Failed to parse bitcode file!");
        }
        LLVM_DEBUG(dbgs() << "\t\t\tlinking into device module...\n");
        if (Linker::linkModules(*LibDeviceModule, std::move(BCModule),
                                Linker::OverrideFromSrc)) {
          errs() << "hipabi transform: device module preloading failed...\n";
          report_fatal_error("hipabi: failed to link device bitcode module!");
        }
      }
    }
    LLVM_DEBUG(dbgs() << "\tfinished rocm bitcode loading+linking.\n");
  }

  return LibDeviceModule;
}

Value *HipABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  // TODO: The grain size on the GPU is a completely different beast than the
  // CPU cases Tapir was originally designed for. At present keeping the grain
  // size at 1 has almost always shown to yield the best results in terms of
  // performance but we should take a closer look...  We have some tweaks for
  // experimenting with this via the command line but it remains unexplored.
  Value *Grainsize;
  Grainsize = ConstantInt::get(GrainsizeCall->getType(), DefaultGrainSize);
  // Replace uses of grain size intrinsic call with a computed
  // grain size value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  GrainsizeCall->eraseFromParent();
  return Grainsize;
}

void HipABI::lowerSync(SyncInst &SI) {
  // no-op
}

void HipABI::addHelperAttributes(Function &F) {
  // no-op
}

bool HipABI::preProcessFunction(Function &F, TaskInfo &TI,
                                bool OutliningTapirLoops) {
  return false;
}

void HipABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {
  // no-op
}

// We can't create a correct launch sequence until all the kernels within a
// (LLVM) module are generated.  When post-processing the module we create the
// fatbinary and then to revisit the kernel launch calls we created at the loop
// level and replace the fat binary pointer/handle with the completed version.
//
// In addition, we must copy data for global variables from the host to the
// device prior to kernel launches.  This requires digging some additional
// details out of the fat binary.
void HipABI::finalizeLaunchCalls(Module &M, GlobalVariable *BundleBin) {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);

  std::vector<CallInst *> LaunchCalls;
  for (Function &Fn : M)
    for (inst_iterator I = inst_begin(Fn); I != inst_end(Fn); ++I)
      if (auto *Call = dyn_cast<CallInst>(&*I))
        if (Function *Callee = Call->getCalledFunction())
          if (Callee->getName().starts_with("__kithip_launch_kernel"))
            LaunchCalls.push_back(Call);

  for (CallInst *Call : LaunchCalls) {
    Value *FatBin = CastInst::CreateBitOrPointerCast(BundleBin, VoidPtrTy,
                                                     "_hipbin.fatbin", Call);
    Call->setArgOperand(0, FatBin);

    // TODO: Do we want to sync naming conventions up between the CUDA and HIP
    // ABIs?  Might make the world a better place???
    for (GlobalVariable *HostGV : GlobalVars) {
      LLVM_DEBUG(dbgs() << "\t\t* processing global: " << HostGV->getName()
                        << "\n");
      // Get the global's name, look it up on the device side, and then issue
      // the copy-to-device call (with appropriate casts).
      std::string DevVarName = HostGV->getName().str() + "_devvar";
      Value* SymName = M.getGlobalVariable(DevVarName);
      if (!SymName)
        SymName = tapir::createConstantStr(DevVarName, M, DevVarName);
      Value *DevPtr = CallInst::Create(
          KitHipGetGlobalSymbolFn, {FatBin, SymName}, ".hipabi_devptr", Call);
      Value *VGVPtr = CastInst::CreatePointerCast(HostGV, VoidPtrTy, "", Call);
      uint64_t NumBytes = DL.getTypeAllocSize(HostGV->getValueType());
      CallInst::Create(KitHipMemcpySymbolToDevFn,
                       {VGVPtr, DevPtr, ConstantInt::get(Int64Ty, NumBytes)},
                       "", Call);
    }
  }

  GlobalVariable *ProxyFB = M.getGlobalVariable(HIPABI_DUMMY_FATBIN_NAME, true);
  if (ProxyFB) {
    Constant *CFB =
        ConstantExpr::getPointerCast(BundleBin, VoidPtrTy->getPointerTo());
    LLVM_DEBUG(dbgs() << "\t\treplacing and removing proxy fatbin ptr.\n");
    ProxyFB->replaceAllUsesWith(CFB);
    ProxyFB->eraseFromParent();
  } else {
    report_fatal_error("unable to find the proxy fatbin pointer! something has "
                       "gone horribly wrong!");
  }
}

HipABIOutputFile HipABI::createTargetObj(const StringRef &ObjFileName) {
  LLVM_DEBUG(dbgs() << "\tgenerating amdgpu object file.\n");

  bool VerboseMode = getOptions().getVerbose();
  std::error_code EC;
  HipABIOutputFile ObjFile = std::make_unique<ToolOutputFile>(
      ObjFileName, EC, sys::fs::OpenFlags::OF_None);
  if (EC) {
    errs() << "hipabi: could not open object file '" << ObjFileName
           << "':" << EC.message();
    report_fatal_error("code transformation failed!");
  }
  ObjFile->keep();

  OptimizationLevel KModOptLevel(Level);
  if (OptLevel != -1) {
    switch (OptLevel) {
    case 0:
      KModOptLevel = OptimizationLevel::O0;
      break;
    case 1:
      KModOptLevel = OptimizationLevel::O1;
      break;
    case 2:
      KModOptLevel = OptimizationLevel::O2;
      break;
    case 3:
      KModOptLevel = OptimizationLevel::O3;
      break;
    default:
      llvm_unreachable("unexpected optimization level!");
    }

    if (VerboseMode)
      errs() << "    - kernel module optimization level: -O"
             << KModOptLevel.getSpeedupLevel() << ".\n";

  } else {
    if (VerboseMode) {
      errs() << "    - matching optimization level with primary pipline: -O"
             << KModOptLevel.getSpeedupLevel() << "\n";
    }
  }

  int SpeedupLevel = KModOptLevel.getSpeedupLevel();
  if (SpeedupLevel > 0) {
    PipelineTuningOptions PTO;

    PTO.LoopUnrolling = SpeedupLevel > 2;
    PTO.LoopInterleaving = SpeedupLevel > 3;
    PTO.LoopStripmine = SpeedupLevel > 2;
    PTO.LoopVectorization = SpeedupLevel > 1;
    PTO.SLPVectorization = SpeedupLevel > 1;

    if (VerboseMode) {
      errs() << "      - loop unrolling: "
             << (PTO.LoopUnrolling ? "yes\n" : "no\n");
      errs() << "      - loop interleaving: "
             << (PTO.LoopInterleaving ? "yes\n" : "no\n");
      errs() << "      - loop stripmine: "
             << (PTO.LoopStripmine ? "yes\n" : "no\n");
      errs() << "      - loop vectorization: "
             << (PTO.LoopVectorization ? "yes\n" : "no\n");
      errs() << "      - slp vectorization: "
             << (PTO.SLPVectorization ? "yes\n" : "no\n");
    }

    // The analysis managers must be declared in this order so that they are
    // destroyed in the correct order due to inter-analysis-manager references.
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    PassBuilder PB(AMDTargetMachine, PTO);
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    AMDTargetMachine->registerPassBuilderCallbacks(PB);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    ModulePassManager MPM =
        PB.buildPerModuleDefaultPipeline(KModOptLevel, false, false);
    MPM.addPass(VerifierPass());
    if (VerboseMode)
      errs() << "    - running kernel/device-side pipeline...\n";
    MPM.run(KernelModule, MAM);
  }

  if (VerboseMode) {
    errs() << "  - emit AMDGPU (target: " << getOptions().getArch()
           << ") object file.\n";
    errs() << "    - file name: " << ObjFile->getFilename() << "\n";
  }
  legacy::PassManager EmitPM;
  if (AMDTargetMachine->addPassesToEmitFile(EmitPM, ObjFile->os(), nullptr,
                                            CodeGenFileType::ObjectFile, false))
    report_fatal_error("hipabi: AMDGPU target failed!");

  EmitPM.run(KernelModule);

  return ObjFile;
}

HipABIOutputFile HipABI::linkTargetObj(const HipABIOutputFile &ObjFile,
                                       const StringRef &LinkedObjFileName) {
  assert(ObjFile != nullptr && "null object file!");

  std::error_code EC;
  HipABIOutputFile LinkedObjFile = std::make_unique<ToolOutputFile>(
      LinkedObjFileName, EC, sys::fs::OpenFlags::OF_None);
  if (EC) {
    errs() << "hipabi: failed to open file '" << LinkedObjFileName
           << "':" << EC.message();
    report_fatal_error("hip code transformation failed!");
  }
  LinkedObjFile->keep();

  // TODO: Pass the path to LLD from the frontend.
  // TODO: The lld invocation below is install prefix and unix-specific...
  ErrorOr<std::string> LLD = sys::findProgramByName("ld.lld");
  if ((EC = LLD.getError()))
    report_fatal_error("executable 'lld' not found! check your path?");

  opt::ArgStringList LLDArgList;
  LLDArgList.push_back(LLD->c_str());
  LLDArgList.push_back("-flavor");
  LLDArgList.push_back("gnu");
  LLDArgList.push_back("-m");
  LLDArgList.push_back("elf64_amdgpu");
  LLDArgList.push_back("--no-undefined");
  LLDArgList.push_back("-shared");
  LLDArgList.push_back("--eh-frame-hdr");
  LLDArgList.push_back("--plugin-opt=-amdgpu-internalize-symbols");
  std::string mcpu = "--plugin-opt=-mcpu=" + getOptions().getArch().str();
  if (EnableXnack)
    mcpu += ":xnack+";
  if (EnableSRAMECC)
    mcpu += ":sramecc+";
  LLDArgList.push_back(mcpu.c_str());

  // TODO: Do we always want this to be -O3, or should this match the "main"
  // optimization level?
  std::string optlevel_arg = "-O" + std::to_string(3);
  LLDArgList.push_back(optlevel_arg.c_str());
  LLDArgList.push_back("-o");
  std::string outfile = LinkedObjFile->getFilename().str();
  LLDArgList.push_back(outfile.c_str());
  std::string infile = ObjFile->getFilename().str();
  LLDArgList.push_back(infile.c_str());
  LLDArgList.push_back(nullptr);

  std::vector<StringRef> LLDArgs = toStringRefArray(LLDArgList.data());
  if (getOptions().getVerbose()) {
    // Render the command line in a single line because some tests expect this.
    // We could consider using a different argument to print just the command
    // lines, but that seems unnecessary.
    //
    // Then render each argument on a separate line because that is easier to
    // parse visually. This comes in handy when debugging.
    errs() << "    - running link stage...\n";
    errs() << "        ";
    tapir::renderCommandLine(LLDArgs, errs());
    errs() << "        $ ";
    for (StringRef Arg : LLDArgList)
      errs() << Arg << "\n          ";
    errs() << "** done.\n";
  }

  std::string ErrMsg;
  bool ExecFailed;
  int ExecStat = sys::ExecuteAndWait(*LLD, LLDArgs, std::nullopt, {},
                                     0, // unlimited wait.
                                     0, // unlimited memory.
                                     &ErrMsg, &ExecFailed);
  if (ExecFailed)
    report_fatal_error("kitsune[hipabi]: 'lld' execution failed!");

  if (ExecStat != 0)
    report_fatal_error("kitsune[hipabi]: 'lld' errors - \n\t" +
                       StringRef(ErrMsg));

  return LinkedObjFile;
}

HipABIOutputFile HipABI::createBundleFile() {
  // At this point the kernel module should have all the necessary pieces from
  // the input module. Convert the kernel module into a fat binary that can be
  // embedded into the host-side module.
  //
  // We attempt to mimic portions of the steps that the hip/clang frontend uses
  // but given we are "mid-stage" there are some differences.
  //
  // TODO: At present this produces working code but the vast majority of tools
  // (e.g., rocm-obj) don't appear to work correctly.

  bool VerboseMode = getOptions().getVerbose();
  std::error_code EC;

  // Run the AMDGPU target to create the associated object file for the
  // kernel module.
  std::string ModelBundleFileName = "%%-%%-%%_" + KernelModule.getName().str();
  SmallString<1024> BundleFileName;
  sys::fs::createUniquePath(ModelBundleFileName.c_str(), BundleFileName, true);
  sys::path::replace_extension(BundleFileName, ".amdgpu.o");
  HipABIOutputFile ObjFile = createTargetObj(BundleFileName.str());
  assert(ObjFile != nullptr && "bad unique ptr!");

  if (VerboseMode)
    errs() << "    - bundle file: " << BundleFileName.str() << "\n";

  // Link the target object file to create a shared object.
  SmallString<255> LinkedObjFileName(BundleFileName);
  sys::path::replace_extension(LinkedObjFileName, ".so");
  HipABIOutputFile LinkedObjFile = linkTargetObj(ObjFile, LinkedObjFileName);

  if (VerboseMode)
    errs() << "    - linked bundle file: " << LinkedObjFileName.str() << "\n";

  if (not KeepIntermediateFiles)
    sys::fs::remove(ObjFile->getFilename());

  LinkedObjFile->keep();
  return LinkedObjFile;
}

GlobalVariable *HipABI::embedBundle(HipABIOutputFile &BundleFile) {
  std::unique_ptr<MemoryBuffer> Bundle = nullptr;
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(BundleFile->getFilename());

  if (std::error_code EC = BufferOrErr.getError()) {
    report_fatal_error("kitsune[hipabi]: failed to load fat binary file: " +
                       StringRef(EC.message()));
  }

  Bundle = std::move(BufferOrErr.get());

  if (getOptions().getVerbose()) {
    errs() << "    - read binary bundle and embed in code.\n";
    errs() << "      - size: " << Bundle->getBufferSize() << " bytes\n";
  }

  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Constant *BundleArray = ConstantDataArray::getRaw(
      StringRef(Bundle->getBufferStart(), Bundle->getBufferSize()),
      Bundle->getBufferSize(), Int8Ty);
  GlobalVariable *BundleGV;
  BundleGV = new GlobalVariable(
      M, BundleArray->getType(), true, GlobalValue::PrivateLinkage, BundleArray,
      KITSUNE_HIP_FATBIN_NAME, nullptr, GlobalVariable::NotThreadLocal);

  const char *BundleSectionName = ".hip_fatbin";
  BundleGV->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
  BundleGV->setSection(BundleSectionName);
  const unsigned HIPCodeObjectAlign = 4096;
  BundleGV->setAlignment(Align(HIPCodeObjectAlign));
  return BundleGV;
}

void HipABI::bindGlobalVariables(Value *FatBinHandle, IRBuilder<> &B) {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *PtrTy = PointerType::getUnqual(Ctx);

  FunctionCallee RegisterVarFn = M.getOrInsertFunction(
      "__hipRegisterManagedVar",
      VoidTy,   // returns nothing...
      PtrTy,    // fatbin handle
      PtrTy,    // Device side (managed) variable.
      PtrTy,    // Global (host side) variable.
      PtrTy,    // variable name (same on both device and host side?)
      Int64Ty,  // variable size (bytes?)
      Int32Ty); // alignment

  for (GlobalVariable *HostGV : GlobalVars) {
    StringRef HostGVName = HostGV->getName();
    uint64_t VarSize = DL.getTypeAllocSize(HostGV->getValueType());
    std::string DevPtrName = HostGVName.str() + ".devptr";
    std::string DevVarName = HostGVName.str() + "_devvar";
    Value* VarName = M.getGlobalVariable(DevVarName);
    if (!VarName)
      VarName = tapir::createConstantStr(DevVarName, M, DevVarName);
    GlobalVariable *DevPtrGV = M.getGlobalVariable(DevPtrName);
    assert(DevPtrGV && "Could not find device pointer global variable");
    LLVM_DEBUG(dbgs() << "\t\thost global '" << HostGVName.str()
                      << "' to device '" << DevVarName << "' using '"
                      << DevPtrName << "'\n");
    Value *Args[] = {FatBinHandle,
                     DevPtrGV,
                     HostGV,
                     VarName,
                     ConstantInt::get(Int64Ty, VarSize),
                     ConstantInt::get(Int32Ty, HostGV->getAlignment())};
    // FIXME: Global variable support on hip is currently broken.
    // This should be fixed, but we have to switch to other things for now.
    // B.CreateCall(RegisterVarFn, Args);
  }
}

Function *HipABI::createCtor(GlobalVariable *Bundle, GlobalVariable *Wrapper) {
  LLVM_DEBUG(dbgs() << "\tcreating global ctor entries...\n");

  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  Type *IntTy = Type::getInt32Ty(Ctx);

  Function *CtorFn = Function::Create(
      FunctionType::get(VoidTy, VoidPtrTy, false), GlobalValue::InternalLinkage,
      tapir::concat(HIPABI_PREFIX, ".ctor.", sys::path::filename(M.getName())),
      &M);

  BasicBlock *CtorEntryBB = BasicBlock::Create(Ctx, "entry", CtorFn);
  IRBuilder<> CtorBuilder(CtorEntryBB);
  const DataLayout &DL = M.getDataLayout();

  FunctionCallee KitRTInitFn =
      M.getOrInsertFunction("__kithip_initialize", VoidTy);
  CtorBuilder.CreateCall(KitRTInitFn, {});

  if (EnableXnack) {
    LLVM_DEBUG(dbgs() << "\t\tenable xnack via ctor runtime call.\n");
    FunctionCallee KitRTEnableXnackFn =
        M.getOrInsertFunction("__kithip_enable_xnack", VoidTy);
    CtorBuilder.CreateCall(KitRTEnableXnackFn, {});
  }

  if (UseYLaunch) {
    LLVM_DEBUG(
        dbgs() << "\t\tenable y-axis launch pattern via ctor runtime call.\n");
    FunctionCallee KitRTEnableYLaunchFn =
        M.getOrInsertFunction("__kithip_enable_ylaunch", VoidTy);
    CtorBuilder.CreateCall(KitRTEnableYLaunchFn, {});
  }

  unsigned DefaultThreadsPerBlock = getOptions().getFixedThreadsPerBlock();
  unsigned MaxThreadsPerBlock = getOptions().getMaxThreadsPerBlock();

  if (DefaultThreadsPerBlock) {
    FunctionCallee KitRTSetDefaultThreadsPerBlockFn =
        M.getOrInsertFunction("__kithip_set_threads_per_blk", VoidTy, IntTy);
    CtorBuilder.CreateCall(KitRTSetDefaultThreadsPerBlockFn,
                           {ConstantInt::get(IntTy, DefaultThreadsPerBlock)});
  }

  FunctionCallee KitRTSetMaxThreadsPerBlockFn =
      M.getOrInsertFunction("__kithip_set_max_threads_per_blk", VoidTy, IntTy);
  if (MaxThreadsPerBlock) {
    CtorBuilder.CreateCall(KitRTSetMaxThreadsPerBlockFn,
                           {ConstantInt::get(IntTy, MaxThreadsPerBlock)});
  } else {
    // If the MaxThreadsPerBlock has not been set, use a value of 1024 anyway.
    // At the time of writing, exceeding this value degrades performance. This
    // might change, and we may even have to set a different value depending
    // on the specific GPU architecture.
    CtorBuilder.CreateCall(KitRTSetMaxThreadsPerBlockFn,
                           {ConstantInt::get(IntTy, 1024)});
  }

  if (getOptions().getRuntimeVerbose()) {
    FunctionCallee KitRTVerboseModefn =
        M.getOrInsertFunction("__kitrt_enable_verbose_mode", VoidTy);
    CtorBuilder.CreateCall(KitRTVerboseModefn, {});
  }

  // TODO: It is still somewhat unclear if we actually need to register fat
  // binaries given we take a different path with codegen here than the more
  // commmon approach done via the frontend (e.g., we have no stub functions).
  // We should dig more into the details to find out if this is actually
  // needed/helpful/etc.  This might mean digging into the ROCm source...
  FunctionCallee RegisterFatbinaryFn =
      M.getOrInsertFunction("__hipRegisterFatBinary",
                            FunctionType::get(VoidPtrPtrTy, VoidPtrTy, false));
  LLVM_DEBUG(dbgs() << "\tregister fat binary.\n");
  CallInst *RegFatbin = CtorBuilder.CreateCall(
      RegisterFatbinaryFn, CtorBuilder.CreateBitCast(Wrapper, VoidPtrTy));
  GlobalVariable *Handle = new GlobalVariable(
      M, VoidPtrPtrTy,
      /*isConstant=*/false, GlobalValue::InternalLinkage,
      ConstantPointerNull::get(VoidPtrPtrTy), "__hip_gpubin_handle");
  Handle->setAlignment(DL.getPointerPrefAlignment());
  CtorBuilder.CreateAlignedStore(RegFatbin, Handle,
                                 DL.getPointerPrefAlignment());

  LoadInst *HandlePtr = CtorBuilder.CreateLoad(
      VoidPtrPtrTy, Handle, tapir::concat(HIPABI_PREFIX, "__hip_fatbin"));
  HandlePtr->setAlignment(DL.getPointerPrefAlignment());

  if (!GlobalVars.empty()) {
    LLVM_DEBUG(dbgs() << "\t\tbinding host and device global variables...\n");
    bindGlobalVariables(HandlePtr, CtorBuilder);
  }

  // Now add a Dtor to help us clean up at program exit...
  if (Function *CleanupFn = createDtor(Handle)) {
    // Hook into 'atexit()'...
    FunctionType *AtExitFnTy =
        FunctionType::get(IntTy, CleanupFn->getType(), false);
    FunctionCallee AtExitFn =
        M.getOrInsertFunction("atexit", AtExitFnTy, AttributeList());

    CtorBuilder.CreateCall(AtExitFn, CleanupFn);
  }

  CtorBuilder.CreateRetVoid();
  return CtorFn;
}

Function *HipABI::createDtor(GlobalVariable *BundleHandle) {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *VoidPtrTy = PointerType::getUnqual(Ctx);
  Type *VoidPtrPtrTy = VoidPtrTy->getPointerTo();

  FunctionCallee UnregisterFatbinFn =
      M.getOrInsertFunction("__hipUnregisterFatBinary",
                            FunctionType::get(VoidTy, VoidPtrPtrTy, false));

  Function *DtorFn = Function::Create(
      FunctionType::get(VoidTy, VoidPtrTy, false), GlobalValue::InternalLinkage,
      tapir::concat(HIPABI_PREFIX, ".dtor"), &M);

  // TODO: Do we call into this too many times???
  BasicBlock *DtorEntryBB = BasicBlock::Create(Ctx, "entry", DtorFn);
  IRBuilder<> DtorBuilder(DtorEntryBB);
  Value *HandleValue = DtorBuilder.CreateAlignedLoad(
      VoidPtrPtrTy, BundleHandle, DL.getPointerABIAlignment(0));
  DtorBuilder.CreateCall(UnregisterFatbinFn, HandleValue);

  // FIXME: There is a bug here which seems to cause use-after-free errors in
  // Kitsune's runtime. It is not entirely clear where exactly the problem is.
  // This causes the kitsune-test-suite to consistently fail. In the interest
  // of having the test suite actually be useful, don't generate the call to
  // __kithip_destroy until we can figure out exactly what is going on there.
  FunctionCallee KitRTDestroyFn =
      M.getOrInsertFunction("__kithip_destroy", VoidTy);
  // DtorBuilder.CreateCall(KitRTDestroyFn, {});
  DtorBuilder.CreateRetVoid();
  return DtorFn;
}

void HipABI::registerBundle(GlobalVariable *Bundle) {
  const int BundleMagicID = 0x48495046;
  const DataLayout &Layout = M.getDataLayout();
  Type *BundleStrTy = Bundle->getType();
  Constant *Zeros[] = {ConstantInt::get(Layout.getIndexType(BundleStrTy), 0),
                       ConstantInt::get(Layout.getIndexType(BundleStrTy), 0)};
  Constant *BundlePtr =
      ConstantExpr::getGetElementPtr(Bundle->getValueType(), Bundle, Zeros);
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  Type *IntTy = Type::getInt32Ty(Ctx);

  StructType *WrapperTy = StructType::get(IntTy,      // magic #
                                          IntTy,      // version
                                          VoidPtrTy,  // binary (gpu executable)
                                          VoidPtrTy); // unused for now.
  Constant *WrapperS =
      ConstantStruct::get(WrapperTy, ConstantInt::get(IntTy, BundleMagicID),
                          ConstantInt::get(IntTy, 1), BundlePtr,
                          ConstantPointerNull::get(VoidPtrTy));
  GlobalVariable *Wrapper =
      new GlobalVariable(M, WrapperTy, true, GlobalValue::InternalLinkage,
                         WrapperS, "__hip_fatbin_wrapper");
  const char *BundleSectionName = ".hipFatBinSegment";
  Wrapper->setSection(BundleSectionName);
  Wrapper->setAlignment(Align(DL.getPrefTypeAlign(Wrapper->getType())));

  Function *CtorFn = createCtor(Bundle, Wrapper);
  if (CtorFn) {
    LLVM_DEBUG(
        dbgs()
        << "\tadding global ctor for runtime and module initialization.\n");
    FunctionType *CtorFnTy = FunctionType::get(VoidTy, false);
    Type *CtorFnPtrTy =
        PointerType::get(CtorFnTy, M.getDataLayout().getProgramAddressSpace());
    tapir::appendToGlobalCtors(M, ConstantExpr::getBitCast(CtorFn, CtorFnPtrTy),
                               65536, nullptr);
  } else
    LLVM_DEBUG(
        dbgs() << "WARNING: received null ctor -- initialization skipped?\n");
}

void HipABI::postProcessModule() {
  bool VerboseMode = getOptions().getVerbose();

  // At this point, all tapir constructs in the input module (M) have been
  // transformed (i.e., outlined) into the kernel module. We can now wrap up
  // module-wide changes for both modules and generate a GPU binary.
  // NOTE: postProcessModule() will not be called in cases where parallelism
  // was not discovered during loop spawning.
  if (VerboseMode) {
    errs() << "kitsune[hipabi]: running kernel module postprocessing "
           << "transformations.\n";
    errs() << "  - kernel module: " << KernelModule.getName() << "\n";
  }

  LLVM_DEBUG(saveModuleToFile(&KernelModule, KernelModule.getName().str(),
                              ".hipabi.preopt.ll"));

  StripDebugInfo(KernelModule);

  if (Function *puts = KernelModule.getFunction("puts")) {
    Value *printf = KernelModule.getFunction("printf");
    if (not printf) {
      LLVMContext &context = KernelModule.getContext();
      Type *paramTys[] = {PointerType::getUnqual(context)};
      Type *retTy = Type::getInt32Ty(context);
      FunctionType *funcTy = FunctionType::get(retTy, paramTys, false);
      FunctionCallee fce = KernelModule.getOrInsertFunction("printf", funcTy);
      printf = fce.getCallee();
    }
    puts->replaceAllUsesWith(printf);
  }

  // Do the final transformation step for the *device* functions in the kernel
  // module. Note that we have already completed the transformations for the
  // outlined loops (the kernel functions) so we skip them here.
  for (Function &F : KernelModule) {
    if (VerboseMode)
      errs() << "    - function: " << F.getName() << "() ";
    if (F.isDeclaration()) {
      if (VerboseMode)
        errs() << "[skipping declaration]\n";
    } else if (isAMDKernelFunction(F)) {
      if (VerboseMode)
        errs() << "[transform kernel]\n";
      transformCallingConv(F);
      transformArguments(F);
      transformConstants(F);
      // KFunc = &F;
    } else {
      if (VerboseMode)
        errs() << "[transform dev-side function]\n";
      // transformForGCN(F); // TODO: deprecated?
      transformCallingConv(F);
      transformConstants(F);
    }
  }

  if (VerboseMode)
    errs() << "  - linking device libraries into kernel...\n";
  linkInModule(getLibDeviceModule());

  // At this point we know all tapir loop constructs in the input
  // module (M) have been processed and the kernel module is populated
  // with the corresponding transformed code and is ready to be
  // converted into a fat binary and then embedded into the host-side
  // module.
  if (VerboseMode)
    errs() << "  - optimze, create and register fat-binary...\n";
  HipABIOutputFile BundleFile = createBundleFile();
  GlobalVariable *Bundle = embedBundle(BundleFile);
  registerBundle(Bundle);

  // Before we finish we now need to patch the launch calls that were
  // initially created before the fat binary was complete.
  if (VerboseMode)
    errs() << "  - bind host-side kernel launches to fat binary...\n";
  finalizeLaunchCalls(M, Bundle);

  LLVM_DEBUG(saveModuleToFile(&KernelModule, KernelModule.getName().str(),
                              ".hipabi.final.ll"));

  // EXPERIMENTAL: We have removed code from the host side and inserted some
  // additional code. Re-run a series of optimization passes -- in general the
  // return on investment here is probably pretty low but we have yet to dig
  // into any details. For now we will only run this at the highest optimization
  // levels.
  if (HostOptLevel > 0) {
    LLVM_DEBUG(dbgs() << "hipabi: Running experimental post-transform "
                      << "host-side (re)optimization passes.\n");

    PipelineTuningOptions pto;
    pto.LoopVectorization = false; // HostOptLevel > 2;
    pto.SLPVectorization = false;  // HostOptLevel > 2;
    pto.LoopUnrolling = HostOptLevel > 1;
    pto.LoopInterleaving = HostOptLevel > 1;
    pto.LoopStripmine = false;
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;
    PassBuilder pb(AMDTargetMachine, pto);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    AMDTargetMachine->registerPassBuilderCallbacks(pb);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    OptimizationLevel optLevels[] = {
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    };

    if (HostOptLevel > 3)
      HostOptLevel = 3;
    OptimizationLevel optLevel = optLevels[HostOptLevel];
    ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(optLevel);
    mpm.addPass(VerifierPass());
    pb.buildPerModuleDefaultPipeline(optLevel);
    mpm.addPass(VerifierPass());
    mpm.run(M, mam);
  }

  if (not KeepIntermediateFiles)
    sys::fs::remove(BundleFile->getFilename());

  if (VerboseMode)
    errs() << "kitsune[hipabi]: kernel module transform complete.\n";
}

LoopOutlineProcessor *
HipABI::getLoopOutlineProcessor(const TapirLoopInfo *TL,
                                OptimizationLevel OptLevel) {

  // Create a HIP loop outline processor for transforming parallel tapir loop
  // constructs into suitable GPU device code.  We hand the outliner the kernel
  // module (KM) as the destination for all generated (device-side) code.
  std::string ModuleName = sys::path::filename(M.getName()).str();
  Loop *TheLoop = TL->getLoop();
  Function *Fn = TheLoop->getHeader()->getParent();
  std::string KernelName = Fn->getName().str();

  if (M.getNamedMetadata("llvm.dbg")) {
    // TODO: Is there any hip specific debug naming?
    // If we have debug info in the module use a line number-based
    // naming scheme for kernels.
    unsigned LineNumber = TL->getLoop()->getStartLoc()->getLine();
    KernelName =
        tapir::concat(HIPABI_KERNEL_NAME_PREFIX, ModuleName, "_", LineNumber);
  } else {
    SmallString<255> ModName(Twine(ModuleName).str());
    sys::path::replace_extension(ModName, "");
    KernelName = tapir::concat(HIPABI_KERNEL_NAME_PREFIX, ModName);
  }

  Level = OptLevel;
  HipLoop *Outliner = new HipLoop(M, KernelModule, KernelName, this);
  return Outliner;
}

void HipABI::pushGlobalVariable(GlobalVariable *GV) {
  GlobalVars.push_back(GV);
}
