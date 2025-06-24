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
// Implementation of Kitsune's cuda tapir target that lowers to Kitsune's cuda
// runtime for NVIDIA GPU's
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/CudaABI.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/KernelProperties.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/TapirTargetOptions.h"
#include "kitsune/Core/TargetUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/TapirLoopHints.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Tapir/TapirGPUUtils.h"
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
//

// Set a specific optimization level for the transformation's pass over the
// created "kernel-module". By default this level will mirror that of the
// frontend but can be set specifically here -- this is primarily useful
// for exploring various details of levels between those operating on the
// Tapir IR and those after the transformation to GPU-friendly LLVM IR.
static cl::opt<int> OptLevel("cuabi-opt-level", cl::init(-1), cl::Hidden,
                             cl::desc("Specify the GPU kernel optimization "
                                      "level. Must be 0, 1, 2 or 3"));

// This is meant to be a factor used for additional kernel optimizations but is
// currently not used this. It should be left in its default state.
static cl::opt<unsigned> DefaultGrainSize(
    "cuabi-default-grainsize", cl::init(1), cl::Hidden,
    cl::desc("The default grain size used by the transform "
             "when analysis fails to determine one (default=1)"));

// Enable/Disable flush denorms-to-zero code generation.
static cl::opt<bool>
    FTZCodeGen("cuabi-ftz", cl::init(false), cl::NotHidden,
               cl::desc("Use flush-denorms-to-zero code generation paths"));

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

// Definition of static member of CudaLoop. See the class declaration for
// documentation of this member.
unsigned CudaLoop::NextKernelID = 0;

CudaLoop::CudaLoop(Module &M, Module &KernelModule, const std::string &KN,
                   const TapirTargetOptions &TTOpts)
    : LoopOutlineProcessor(M, KernelModule, TTOpts,
                           CloneFunctionChangeType::DifferentModule),
      KernelName(KN), KernelModule(KernelModule) {
  KernelName = join_items("", KernelName, "_", std::to_string(NextKernelID));
  NextKernelID++;

  LLVM_DEBUG(dbgs() << "debug[cuabi]: creating a cuda loop outliner.\n"
                    << "  - target kernel name: " << KernelName << "\n");

  // Thread index values -- equivalent to Cuda's builtins:  threadIdx.[x,y,z].
  CUThreadIdxX = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_tid_x);
  CUThreadIdxY = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_tid_y);
  CUThreadIdxZ = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_tid_z);

  // Block index values -- equivalent to Cuda's builtins: blockIndx.[x,y,z].
  CUBlockIdxX = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
  CUBlockIdxY = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ctaid_y);
  CUBlockIdxZ = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ctaid_z);

  // Block dimensions -- equivalent to Cuda's builtins: blockDim.[x,y,z].
  CUBlockDimX = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ntid_x);
  CUBlockDimY = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ntid_y);
  CUBlockDimZ = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ntid_x);

  // Grid dimensions -- equivalent to Cuda's builtins: gridDim.[x,y,z].
  CUGridDimX = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_nctaid_x);
  CUGridDimY = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_nctaid_y);
  CUGridDimZ = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_nctaid_z);
}

CudaLoop::~CudaLoop() {
  LLVM_DEBUG(dbgs() << "debug[cuabi]: destroying loop outliner for kernel '"
                    << KernelName << "'.\n");
}

void CudaLoop::setupLoopOutlineArgs(Function &F, ValueSet &HelperArgs,
                                    SmallVectorImpl<Value *> &HelperInputs,
                                    ValueSet &InputSet,
                                    const SmallVectorImpl<Value *> &LCArgs,
                                    const SmallVectorImpl<Value *> &LCInputs,
                                    const ValueSet &TLInputsFixed) {
  LLVM_DEBUG(dbgs() << "debug[cuabi]: setting up loop outline arguments...\n");

  // Add the loop control inputs -- the first parameter defines the extent of
  // the index space (the number of threads to launch).
  {
    Argument *EndArg = cast<Argument>(LCArgs[1]);
    EndArg->setName("runSize");
    HelperArgs.insert(EndArg);

    Value *InputVal = LCInputs[1];
    HelperInputs.push_back(InputVal);
    InputSet.insert(InputVal);
  }

  // The second parameter defines the start of the index space.
  {
    Argument *StartArg = cast<Argument>(LCArgs[0]);
    StartArg->setName("runStart");
    HelperArgs.insert(StartArg);

    Value *InputVal = LCInputs[0];
    HelperInputs.push_back(InputVal);
    InputSet.insert(InputVal);
  }

  // The third parameter defines the grain size, if it is not constant.
  if (!isa<ConstantInt>(LCInputs[2])) {
    Argument *GrainsizeArg = cast<Argument>(LCArgs[2]);
    GrainsizeArg->setName("grainSize");
    HelperArgs.insert(GrainsizeArg);

    Value *InputVal = LCInputs[2];
    HelperInputs.push_back(InputVal);
    InputSet.insert(InputVal);
  }

  // Add the loop-centric kernel parameters (i.e., variables/arrays
  // used in the loop body).
  LLVM_DEBUG(dbgs() << "  - adding loop-centric kernel arguments...\n");
  for (Value *V : TLInputsFixed) {
    HelperArgs.insert(V);
    HelperInputs.push_back(V);
    LLVM_DEBUG(dbgs() << "    - arg: " << V->getName() << "\n");
  }

  LLVM_DEBUG(dbgs() << "  - adding helper kernel arguments...\n");
  for (Value *V : HelperInputs) {
    OrderedInputs.push_back(V);
    LLVM_DEBUG(dbgs() << "    - helper arg: " << V->getName() << "\n");
  }

  LLVM_DEBUG(dbgs() << "  - done.\n");
}

unsigned CudaLoop::getIVArgIndex(const Function &F,
                                 const ValueSet &Args) const {
  // The argument for the primary induction variable is the second input.
  return 1;
}

unsigned CudaLoop::getLimitArgIndex(const Function &F,
                                    const ValueSet &Args) const {
  // The argument for the loop limit is the first input.
  return 0;
}

void CudaLoop::preProcessTapirLoop(TapirLoopInfo &TL, ValueToValueMapTy &VMap) {
  LLVM_DEBUG(dbgs() << "debug[cuabi]: -preprocessing loop for kernel '"
                    << KernelName << "'.\n");

  // Collect the top-level entities (Function, GlobalVariable, GlobalAlias
  // and GlobalIFunc) that are used in the outlined loop. Since the outlined
  // loop will live in the KernelModule, any GlobalValue's used in it must be
  // be cloned into the KernelModule and then registered with the cuda runtime.
  // The registration will be done in he global ctor.
  LLVM_DEBUG(dbgs() << "  - gathering and analyzing global values...\n");
  collectGlobalValues(*TL.getLoop(), UsedGlobalValues);

  // FIXME: Support GlobalIFunc at some point. This is a GNU extension, so we
  // may not want to support it at all, but just in case, this is here. We
  // probably do want to support GlobalAlias at some point, but we defer it for
  // the moment since we have a number of other things to support first.
  for (GlobalValue *V : UsedGlobalValues)
    if (isa<GlobalIFunc>(V))
      llvm_unreachable("cuabi: NOT YET IMPLEMENTED: GlobalIFunc");
    else if (isa<GlobalAlias>(V))
      llvm_unreachable("cuabi: NOT YET IMPLEMENTED: GlobalAlias");

  // Clone the global variables and aliases first. We probably don't need to it
  // strictly in this order, but later in the code we do, so try to be symmetric
  // here just in case.
  LLVM_DEBUG(dbgs() << "  - cloning global variables into kernel module...\n");
  for (GlobalValue *V : UsedGlobalValues) {
    if (auto *GV = dyn_cast<GlobalVariable>(V)) {
      // It would be nice to do this as a "post-processing" pass, for instance,
      // when preparing the kernel module for PTX. However, we need to pass the
      // name of the global as a string to several Kitsune intrinsics. That
      // would make the job of renaming them later much more complicated since
      // we would have to modify all the calls in the host that use the global
      // variable's name.
      std::string GVName = convertNameForPTX(GV->getName());
      bool IsConstant = GV->isConstant();
      Type *GVType = GV->getValueType();
      GlobalValue::ThreadLocalMode ThreadLocalMode = GV->getThreadLocalMode();
      unsigned AddrSpace = GV->getType()->getAddressSpace();

      GlobalVariable *NewGV = nullptr;
      if ((NewGV =
               KernelModule.getGlobalVariable(GVName, /*AllowLocal=*/true))) {
        // If a global with the name is already present in the kernel module,
        // another outlined loop in the host module used the same global. The
        // global is already present, so we just need to update VMap correctly.
        // This is done after this if-else block.
      } else if (IsConstant) {
        // If the global variable is a constant we can clone it into the device
        // module along with its initializer where it will be treated as an
        // internal variable. There is no coordination with the host.
        NewGV = new GlobalVariable(
            KernelModule, GVType, IsConstant, GlobalValue::InternalLinkage,
            GV->getInitializer(), GVName,
            /*InsertBefore=*/nullptr, ThreadLocalMode, AddrSpace);
        NewGV->setDSOLocal(true);
        NewGV->setAlignment(GV->getAlign());
        LLVM_DEBUG(dbgs() << "    - new constant global variable: '"
                          << NewGV->getName() << "', from '" << GV->getName()
                          << "'.\n");
      } else {
        // If the global is not constant, we will need to create a device-side
        // version that will have the host-side value copied over prior to
        // launching the kernel.
        NewGV = new GlobalVariable(
            KernelModule, GVType, IsConstant, GlobalValue::ExternalLinkage,
            Constant::getNullValue(GVType), GVName,
            /*InsertBefore=*/nullptr, ThreadLocalMode, AddrSpace);
        NewGV->setDSOLocal(true);
        NewGV->setAlignment(GV->getAlign());
        LLVM_DEBUG(dbgs() << "\t\t\tcreated kernel-side global variable '"
                          << NewGV->getName() << "'.\n");
      }
      assert(NewGV && "All global variables must have a corresponding global "
                      "in the kernel module");
      VMap[GV] = NewGV;
    }
  }

  // Functions that are called from the tapir loop must be cloned into the
  // kernel module, especially if they contain a body. This is a two-step
  // process - first we create a declaration for the functions since these may
  // be called by the other device functions. The VMap already contains mappings
  // for the global variables that may be needed
  for (GlobalValue *G : UsedGlobalValues) {
    if (auto *F = dyn_cast<Function>(G)) {
      Function *DeviceF = KernelModule.getFunction(F->getName());
      if (not DeviceF) {
        DeviceF = Function::Create(F->getFunctionType(), F->getLinkage(),
                                   F->getName(), KernelModule);
        for (unsigned I = 0; I < F->arg_size(); ++I) {
          Argument *Arg = F->getArg(I);
          Argument *ArgDev = DeviceF->getArg(I);
          ArgDev->setName(Arg->getName());
          VMap[Arg] = ArgDev;
        }
        LLVM_DEBUG(dbgs() << "\tcreated device-side function declaration for '"
                          << F->getName() << "()'.\n");
      }
      VMap[F] = DeviceF;
    }
  }

  // Now clone any function bodies that need to be cloned.
  for (GlobalValue *V : UsedGlobalValues) {
    if (Function *F = dyn_cast<Function>(V)) {
      if (F->size() && not F->isIntrinsic()) {
        SmallVector<ReturnInst *, 8> Returns;
        Function *DeviceF = cast<Function>(VMap[F]);
        CloneFunctionInto(DeviceF, F, VMap,
                          CloneFunctionChangeType::DifferentModule, Returns);
        DeviceF->addFnAttr(Attribute::KitDevice);
        LLVM_DEBUG(dbgs() << "cuabi: cloning device function '"
                          << DeviceF->getName() << "' into kernel module.\n");
      }
    }
  }
}

void CudaLoop::postProcessOutline(TapirLoopInfo &TLI, TaskOutlineInfo &Out,
                                  ValueToValueMapTy &VMap) {
  LLVMContext &Ctx = M.getContext();
  Task *T = TLI.getTask();
  Loop *TL = TLI.getLoop();

  TapirLoopHints Hints(TL);

  BasicBlock *Entry = cast<BasicBlock>(VMap[TL->getLoopPreheader()]);
  BasicBlock *Header = cast<BasicBlock>(VMap[TL->getHeader()]);
  BasicBlock *Exit = cast<BasicBlock>(VMap[TLI.getExitBlock()]);
  PHINode *PrimaryIV = cast<PHINode>(VMap[TLI.getPrimaryInduction().first]);
  Value *PrimaryIVInput = PrimaryIV->getIncomingValueForBlock(Entry);
  Type *PrimaryIVType = PrimaryIV->getType();

  // We no longer need the cloned sync region.
  auto *ClonedSyncReg =
      cast<Instruction>(VMap[T->getDetach()->getSyncRegion()]);
  ClonedSyncReg->eraseFromParent();

  Function *KernelF = Out.Outline;

  // Set the generated name for the kernel. This name is passed to the runtime's
  // kernel launch function, so it must be set correctly.
  KernelF->setName(KernelName);

  // Set the linkage of the kernel to external to prevent it from being DCE'ed
  // since there will be no caller for the function in the kernel module.
  KernelF->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);

  // Remove all target-related attributes from the kernel function. These may be
  // present because the frontend believes that the code is being compiled for
  // the CPU (host) only.
  KernelF->removeFnAttr("target-cpu");
  KernelF->removeFnAttr("target-features");
  KernelF->removeFnAttr("tune-cpu");

  // Remove some functions that are relevant for functionality that is not
  // supported on the GPU. For instance, exceptions are not currently available
  // on GPU's.
  KernelF->removeFnAttr("personality");

  // Add an attribute identifying this as a function outlined from a tapir loop.
  KernelF->addFnAttr(Attribute::KitKernel);

  // Replace some of the target-specific attributes with the correct ones.
  KernelF->addFnAttr("target-cpu", getOptions().getCudaArch());
  KernelF->addFnAttr("target-features",
                     join_items(",", getOptions().getCudaTargetFeatures(),
                                getOptions().getCudaArch()));

  // Add other attributes that are relevant for the target.
  KernelF->addFnAttr("uniform-work-group-size", "true");

  NamedMDNode *Annotations =
      KernelModule.getOrInsertNamedMetadata("nvvm.annotations");
  SmallVector<Metadata *, 6> AV;
  AV.push_back(ValueAsMetadata::get(KernelF));
  AV.push_back(MDString::get(Ctx, "kernel"));
  AV.push_back(
      ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 1)));
  // AV.push_back(MDString::get(Ctx, "maxntidx"));
  // AV.push_back(ValueAsMetadata::get(
  //     ConstantInt::get(Type::getInt32Ty(Ctx), MaxThreadsPerBlock)));
  Annotations->addOperand(MDNode::get(Ctx, AV));

  // Verify that the Thread ID corresponds to a valid iteration. Because Tapir
  // loops use canonical induction variables, valid iterations range from 0 to
  // the loop limit with stride 1. The End argument encodes the loop limit. Get
  // end and grainsize arguments
  Argument *End;
  Value *Grainsize;
  {
    // TODO: We only support a grain size of 1 right now. Not clear if this
    // could be a future optimization but strip mining on our current tests only
    // results in degraded performance...
    auto OutlineArgsIter = KernelF->arg_begin();
    // End argument is the first LC arg.
    End = &*OutlineArgsIter++;

    // Get the grainsize value, which is either constant or the third LC arg.
    // if (unsigned ConstGrainsize = TLI.getGrainsize())
    //  Grainsize = ConstantInt::get(PrimaryIV->getType(), ConstGrainsize);
    // else
    Grainsize =
        ConstantInt::get(PrimaryIV->getType(), DefaultGrainSize.getValue());
  }

  IRBuilder<> B(Entry->getTerminator());

  // Get the thread ID for this invocation of Helper.
  //
  // This is the classic CUDA thread ID calculation:
  //      i = blockDim.x * blockIdx.x + threadIdx.x;
  // For now we only generate 1-D thread IDs.
  Value *ThreadIdx = B.CreateCall(CUThreadIdxX);
  Value *BlockIdx = B.CreateCall(CUBlockIdxX);
  Value *BlockDim = B.CreateCall(CUBlockDimX);
  Value *BDxBI = B.CreateMul(BlockIdx, BlockDim, "blk_offset");
  Value *TIpBDxBI = B.CreateAdd(ThreadIdx, BDxBI, "cuthread_id");
  Value *ThreadIV =
      B.CreateIntCast(TIpBDxBI, PrimaryIVType, false, "thread_iv");

  // NOTE/TODO: Assuming that the grainsize is fixed at 1 for the current
  // codegen.
  // ThreadID = B.CreateMul(ThreadID, Grainsize);
  Value *ThreadEnd = B.CreateAdd(ThreadIV, Grainsize, "thread_end");
  Value *Cond = B.CreateICmpUGE(ThreadIV, End, "cond_thread_end");
  ReplaceInstWithInst(Entry->getTerminator(),
                      BranchInst::Create(Exit, Header, Cond));

  // Use the thread ID as the start iteration number for the primary IV.
  PrimaryIVInput->replaceAllUsesWith(ThreadIV);
  // TODO: ???? PrimaryIVInput->eraseFromParent();

  // Update cloned loop condition to use the thread-end value.
  unsigned TripCountIdx = 0;
  ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TLI.getCondition()]);
  if (ClonedCond->getOperand(0) != End)
    ++TripCountIdx;
  assert(ClonedCond->getOperand(TripCountIdx) == End &&
         "End argument not used in condition!");
  ClonedCond->setOperand(TripCountIdx, ThreadEnd);
}

void CudaLoop::remapData(ValueToValueMapTy &VMap) {
  for (auto &V : OrderedInputs) {
    if (auto MappedV = VMap[V]) {
      V = MappedV;
    }
  }
}

void CudaLoop::processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                                       DominatorTree &DT) {
  LLVM_DEBUG(dbgs() << "cudaloop: processing outlined loop call...\n"
                    << "\tkernel name: " << KernelName << "\n");

  const DataLayout &DL = M.getDataLayout();
  LLVMContext &Ctx = M.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  PointerType *PtrTy = PointerType::getUnqual(Ctx);
  ArrayType *ArrayTy = ArrayType::get(PtrTy, OrderedInputs.size());

  Function *KitrtSymbolDevicePtr =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::kitrt_symbol_device_ptr);
  Function *KitrtSymbolMemcpyDevice = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::kitrt_symbol_memcpy_device);
  Function *KitrtSymbolMemcpyHost = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::kitrt_symbol_memcpy_host);
  Function *KitrtPrefetchDevice =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::kitrt_prefetch_device);

  // NOTE: If we are dealing with loop nests with multiple targets (in this case
  // only a CPU-target w/ a nested GPU target is supported) we can end up with
  // multiple calls to the outlined loop (which has been setup for dead code
  // elimination) but can cause invalid IR that trips us up when handling the
  // GPU module code generation. This is a challenge in the Tapir design that
  // was not geared to handle some of the nuances of GPU target transformations
  // (and code gen).  To address this, we need to do some clean up to keep the
  // IR correct (or the verifier will fail on us...). Specifically, we can no
  // longer depend upon DCE as it runs too late in the GPU transformation
  // process...
  //
  // TODO: This code can be shared between the cuda and hip targets...
  Function *TargetKF = KernelModule.getFunction(KernelName);
  std::list<Instruction *> RemoveList;
  if (TargetKF) {
    LLVM_DEBUG(dbgs() << "\t*- searching for 'dangling' outline calls...\n");
    for (Use &U : TargetKF->uses()) {
      if (auto *Inst = dyn_cast<Instruction>(U.getUser())) {
        LLVM_DEBUG(dbgs() << "\t\t- marking use for removal.\n");
        if (Inst != TOI.ReplCall)
          RemoveList.push_back(Inst);
      }
    }
  }
  for (Instruction *I : RemoveList)
    I->eraseFromParent();

  ConstantInt *ConstTT = createConstInt(TTID::Cuda, Ctx);
  Value *CudaStream = ConstantPointerNull::get(PtrTy);
  GlobalVariable *InstMix = createKernelPropertiesGlobal(KernelName, M);
  Value *KName = createConstString(KernelName, M);
  GlobalVariable *EmbFB = getEmbFBGlobal(TTID::Cuda, M);

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
  Value *TPB = ConstantInt::get(Int32Ty, Hints.getThreadsPerBlock());

  // Create two builders
  //
  //   1. EntryBuilder: Inserts instructions - typically allocas into the entry
  //      block
  //   2. NewBuilder: Inserts new code in a split basic block
  //
  // *** NOTE: If you are going to code gen an alloca in the code below it is
  // most likely (100%?) you should use the EntryBuilder vs. the NewBuilder. If
  // you find yourself with stack issues for longer running code this is a
  // likely bug to check.
  Function *Parent = TOI.ReplCall->getFunction();
  BasicBlock &EntryBB = Parent->getEntryBlock();
  IRBuilder<> EntryBuilder(&EntryBB.front());

  BasicBlock *RCBB = TOI.ReplCall->getParent();
  BasicBlock *NewBB = RCBB->splitBasicBlock(TOI.ReplCall);
  IRBuilder<> NewBuilder(&NewBB->front());

  // Deal with type mismatches for the trip count.
  Value *TripCount = OrderedInputs[0];
  if (TripCount->getType() != Int64Ty)
    TripCount = NewBuilder.CreateSExtOrBitCast(TripCount, Int64Ty, "cast.tc");

  std::map<GlobalVariable *, Value *> DevGlobals;
  for (GlobalValue *G : UsedGlobalValues) {
    if (auto *GV = dyn_cast<GlobalVariable>(G)) {
      if (not GV->isConstant()) {
        Value *SymName = createConstString(GV->getName(), M);
        DevGlobals.emplace(GV, SymName);
      }
    }
  }

  // We need to explicitly add code to sync up host-side and device-side globals
  // prior to launching kernels.
  for (auto [DevGV, SymName] : DevGlobals) {
    LLVM_DEBUG(dbgs() << "\t\t\t  processing global: '" << DevGV->getName()
                      << "'\n");
    StringRef GVName = DevGV->getName();
    GlobalVariable *HostGV = M.getGlobalVariable(GVName, /*AllowLocal=*/true);
    Type *GVType = DevGV->getValueType();
    size_t GVSize = DL.getTypeAllocSize(GVType);
    Value *DevPtr =
        NewBuilder.CreateCall(KitrtSymbolDevicePtr, {ConstTT, EmbFB, SymName});
    Constant *Bytes = ConstantInt::get(Int64Ty, GVSize);
    NewBuilder.CreateCall(KitrtSymbolMemcpyDevice,
                          {ConstTT, DevPtr, HostGV, Bytes});
  }

  // TODO: There is some potential here to share this code across both the hip
  // and cuda tapir targets.
  LLVM_DEBUG(dbgs() << "\t*- code gen packing of " << OrderedInputs.size()
                    << " kernel args.\n");
  Value *ArgArray = EntryBuilder.CreateAlloca(ArrayTy);

  for (size_t I = 0; I < OrderedInputs.size(); ++I) {
    Value *Inp = OrderedInputs[I];
    Value *InpAlloca = EntryBuilder.CreateAlloca(Inp->getType());
    NewBuilder.CreateStore(Inp, InpAlloca);

    Value *ArgPtr =
        NewBuilder.CreateConstInBoundsGEP2_32(ArrayTy, ArgArray, 0, I);
    NewBuilder.CreateStore(InpAlloca, ArgPtr);

    if (getOptions().getGPUPrefetch() && Inp->getType()->isPointerTy()) {
      LLVM_DEBUG(dbgs() << "\t\t- code gen prefetch for kernel arg #" << I
                        << "\n");
      // The pointer to the data to be prefetched must point to UVM allocated
      // memory. By setting the number of bytes to be prefetched to -1, we are
      // instructing the runtime to prefetch the entire UVM-allocated buffer.
      // The runtime keeps track of this.
      //
      // TODO: Do some analysis to only prefetch the number of bytes that are
      // actually used (or likely to be used) by the kernel.
      ConstantInt *Bytes = NewBuilder.getInt64(-1);
      CudaStream = NewBuilder.CreateCall(KitrtPrefetchDevice,
                                         {ConstTT, Inp, Bytes, CudaStream});
    }
  }

  Value *Args = NewBuilder.CreateConstInBoundsGEP2_32(ArrayTy, ArgArray, 0, 0);

  // TODO: We should probably have the launch and sync kitsune intrinsics take
  // a sync region as an argument This may make it easier to do post-outlining
  // analyses to eliminate/delay device synchronization calls instead of
  // always synchronizing immediately after the kernel launch.
  LLVM_DEBUG(dbgs() << "\t*- code gen kernel launch....\n");
  CudaStream = NewBuilder.CreateCall(
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::kitrt_launch_kernel),
      {ConstTT, EmbFB, KName, Args, TripCount, TPB, InstMix, CudaStream});

  NewBuilder.CreateCall(
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::kitrt_sync_stream),
      {ConstTT, CudaStream});

  // After the kernel is done, copy the non-const globals back to the host. This
  // is done here to keep this part of the code generation simple. A subsequent
  // pass will attempt to move this call to the point where the global is
  // actually used on the host (or perhaps even delete it if the host never uses
  // the global again).
  for (auto [DevGV, SymName] : DevGlobals) {
    LLVM_DEBUG(dbgs() << "\t\t\t  processing global: '" << DevGV->getName()
                      << "'\n");
    StringRef GVName = DevGV->getName();
    GlobalVariable *HostGV = M.getGlobalVariable(GVName, /*AllowLocal=*/true);
    Type *GVType = DevGV->getValueType();
    size_t GVSize = DL.getTypeAllocSize(GVType);
    Value *DevPtr =
        NewBuilder.CreateCall(KitrtSymbolDevicePtr, {ConstTT, EmbFB, SymName});
    Constant *Bytes = ConstantInt::get(Int64Ty, GVSize);
    NewBuilder.CreateCall(KitrtSymbolMemcpyHost,
                          {ConstTT, HostGV, DevPtr, Bytes});
  }

  TOI.ReplCall->eraseFromParent();
  LLVM_DEBUG(dbgs() << "*** finished processing outlined call.\n");
}

CudaABI::CudaABI(Module &M, const TapirTargetOptions &TTO)
    : TapirTarget(M, TTO), KernelModule("", M.getContext()) {
  LLVM_DEBUG(dbgs() << "cuabi: CudaABI::CudaABI()\n");

  // This is used for testing, so it should not be removed. At some point, this
  // will be replaced with something less stupid for testing.
  if (TTO.getTapirVerbose())
    TTO.print(dbgs());

  TargetMachine *TM = createTargetMachine(TTID::Cuda, TTO);
  KernelModule.setModuleIdentifier(
      join_items("", CUABI_PREFIX, sys::path::filename(M.getName())));
  KernelModule.setTargetTriple(TM->getTargetTriple().str());
  KernelModule.setDataLayout(TM->createDataLayout());
  KernelModule.setModuleFlag(Module::Override, "nvvm-reflect-ftz", FTZCodeGen);
  addDeviceModuleMetadata(TTID::Cuda, KernelModule);
}

CudaABI::~CudaABI() { LLVM_DEBUG(dbgs() << "cuabi: destroy tapir target.\n"); }

Value *CudaABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  // TODO: The grainsize on the GPU is a completely different beast than the CPU
  // cases Tapir was originally designed for. At present keeping the grainsize
  // at 1 has almost always shown to yield the best results.  It is obviously
  // not the best choice for all cases...
  Value *Grainsize =
      ConstantInt::get(GrainsizeCall->getType(), DefaultGrainSize.getValue());
  // Replace uses of grainsize intrinsic call with a computed grainsize value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  GrainsizeCall->eraseFromParent();
  return Grainsize;
}

void CudaABI::lowerSync(SyncInst &SI) {
  // The CUDA transformation splits the code into two modules, one for the host,
  // the other for the device. The sync instruction will only be present on the
  // host module.
}

void CudaABI::addHelperAttributes(Function &F) {}

void CudaABI::preProcessModule() {
  // Create the global variable that will eventually contain the fat binary of
  // GPU code. This is currently uninitialized, but will be passed to several
  // of the kitsune runtime intrinsic calls when launching kernels, copying
  // global variables from host to device etc.
  (void)createEmbFBGlobal(TTID::Cuda, M);
}

bool CudaABI::preProcessFunction(Function &F, TaskInfo &TI,
                                 bool OutliningTapirLoops) {
  return false;
}

void CudaABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {}

void CudaABI::postProcessHelper(Function &F) {}

void CudaABI::preProcessOutlinedTask(Function &, Instruction *, Instruction *,
                                     bool, BasicBlock *) {}

void CudaABI::postProcessOutlinedTask(Function &F, Instruction *DetachPt,
                                      Instruction *TaskFrameCreate,
                                      bool IsSpawner, BasicBlock *TFEntry) {}

void CudaABI::postProcessRootSpawner(Function &F, BasicBlock *TFEntry) {}

void CudaABI::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) {}

void CudaABI::preProcessRootSpawner(Function &, BasicBlock *TFEntry) {}

void CudaABI::postProcessModule() {
  LLVM_DEBUG(dbgs() << "cuabi: post processing kernel and host modules...\n");
  LLVM_DEBUG(saveModuleToFile(&KernelModule,
                              M.getName().str() + ".kmod.pre-postproc"));

  // TODO #1: Need to do some more work on debugging and debug info...
  // Make sure any outlined (cloned) debugged info is removed from the kernel
  // module (if we don't it will show up duplicated w/ the host-side module).
  StripDebugInfo(KernelModule);

  // At this point, we are done with the minimum task of outlining the tapir
  // loop into a kernel module. There are still a number of transformations that
  // must be carried out on this module before it can be compiled to GPU code,
  // but those will be done by subsequent passes. The module here is in a state
  // where we can perform combined host/device analyses and optimizations.
  (void)createEmbBCGlobal(KernelModule, TTID::Cuda, M);
}

LoopOutlineProcessor *
CudaABI::getLoopOutlineProcessor(const TapirLoopInfo *TL) {
  LLVM_DEBUG(dbgs() << "cuabi: create loop outlining processor.\n");
  LLVM_DEBUG(saveModuleToFile(&M, M.getName().str() + ".input"));

  std::string KernelName;
  raw_string_ostream Os(KernelName);
  Os << CUABI_KERNEL_NAME_PREFIX;

  // TODO #1: Need to do some more work on debugging and debug info.
  const Loop *Loop = TL->getLoop();
  if (M.getNamedMetadata("llvm.dbg.cu") || M.getNamedMetadata("llvm.dbg")) {
    // If we have debug info in the module use the line number to name the
    // kernel. This is only to make debugging a shade easier since it makes it
    // easier to associate the kernel function with a loop in source code.
    //
    // FIXME: This is risky. In principle, in a large project, we could have
    // multiple files with the same name in different directories. There is a
    // small possibility that a forall loop occurs on exactly the same line in
    // both of these files. Ideally, we should include the full file path which
    // is guaranteed to be unique. However, that would detract from the
    // "usefulness" of this name (mainly for debugging). For now, we'll stick
    // with this until we can make some of the support tooling more robust to
    // allow us to mangle the name to avoid collisions.
    DebugLoc Loc = Loop->getStartLoc();
    unsigned Line = Loc.getLine();
    unsigned Col = Loc.getCol();
    StringRef FilePath = Loc->getFile()->getFilename();
    StringRef FileName = sys::path::filename(FilePath);
    Os << convertNameForPTX(FileName, false) << "_" << Line << "_" << Col;
  } else {
    Function *F = Loop->getHeader()->getParent();
    StringRef FName = F->getName();
    std::string DemangledName;
    if (nonMicrosoftDemangle(FName, DemangledName,
                             /*CanHaveLeadingDot=*/false,
                             /*ParseParams=*/false))
      Os << DemangledName;
    else
      Os << FName;
  }
  LLVM_DEBUG(dbgs() << "\t- kernel function '" << KernelName << "()'.\n");

  return new CudaLoop(M, KernelModule, KernelName, this->getOptions());
}
