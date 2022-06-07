//===- CudaABI.cpp - Lower Tapir to the Kitsune GPU back end -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Kitsune+Tapir GPU ABI to convert Tapir
// instructions to call into the Cuda-centric portions of the Kitsune
// runtime system for GPUs (bypassing the top-level platform independent API
// that is a JIT-only interface).
//
// TODO:
//
//    1. Functionality with CUDA toolchain components is only partially
//       working.  In particular, only one kernel per executable is visible
//       with `cuobjdump`.  Should we be adding multiple binaries to the
//       same handle vs. creating multiple handles?  Code does seem to
//       execute correctly but structure is likely incorrect for tool
//       support.  For more specific details on the tools look here:
//
//          https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
//
//       It is worth doing some poking around...  The fatbinary struct
//       is
//
//         struct fatBinWrapper {
//           int magic;
//           int version;
//           const unsigned long long* data;
//           void *filename_or_fatbins;  /* version 1: offline filename,
//                                          version 2: array of prelinked
//                                          fatbins */
//         };
//
//       This certainly suggests we want to append fatbins at the end of
//       struct and not create a new one every time!
//
//    2. Global variable registration and handling is broken; currently
//       via a invalid CUDA context at runtime.  Could be a mixture of
//       CUDA and driver runtime use -- #1 becomes worse if we don't
//       register the fat binary (do correct driver interface calls
//       exist?).
//
//===----------------------------------------------------------------------===//
#include "llvm/Transforms/Tapir/CudaABI.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Vectorize.h"

#include <fstream>
#include <sstream>

using namespace llvm;

#define DEBUG_TYPE "cuda-abi"

// Some default naming convensions.
static const std::string CUABI_KERNEL_NAME_PREFIX = "__cuabi_kern_";
static const std::string CUABI_MODULE_NAME_PREFIX = "__cuabi_mod_";
static constexpr unsigned FATBINARY_MAGIC_ID = 0x466243b1;

/// Selected target GPU architecture --  passed directly to ptxas.
static cl::opt<std::string>
    GPUArch("cuabi-arch", cl::init("sm_75"), cl::NotHidden,
            cl::desc("Target GPU architecture for CUDA ABI transformation."
                     "(default: sm_75)"));
/// Select 32- vs. 64-bit host architecture. Passed directly to ptxas.
static cl::opt<std::string>
    HostMArch("cuabi-march", cl::init("64"), cl::NotHidden,
              cl::desc("Specify 32- or 64-bit host architecture."
                       "(default=64-bit)."));
/// Enable verbose mode.  Handled internally as well as passed on to
/// ptxas.
static cl::opt<bool>
    Verbose("cuabi-verbose", cl::init(false), cl::NotHidden,
            cl::desc("Enable verbose mode and also print out code "
                     "generation statistics. (default=off)"));
/// Enable debug mode. Passed directly to ptxas.
static cl::opt<bool>
    Debug("cuabi-debug", cl::init(false), cl::NotHidden,
          cl::desc("Enable debug information for GPU device code. "
                   "(default=false)"));
/// Surpress the generation of debug sections in the final object
/// file.  This option is ignored if not used with the debug or
/// generate-line-info options.
static cl::opt<bool>
    SurpressDBInfo("cuabi-surpress-debug-info", cl::init(false), cl::Hidden,
                   cl::desc(" Do not generate debug information sections "
                            "in final output object file."));
/// Generate line information for all generated GPU kernels.  Passed
/// directly to 'ptxas' as '--generate-line-info'.
static cl::opt<bool>
    GenLineInfo("cuabi-generate-line-info", cl::init(false), cl::NotHidden,
                cl::desc("Generate line information for generated GPU "
                         "kernels. (default=false)"));
/// Enable stack bounds checking. Passed directly to 'ptxas', which
/// will automatically turn this on if 'device-debug' or 'opt-level=0'
/// is used.
static cl::opt<bool>
    BoundsCheck("cuabi-bounds-check", cl::init(false), cl::NotHidden,
                cl::desc("Enable GPU static pointer bounds checking. "
                         "(default=false)"));
/// Set the optimization level for the compiler.  Values can be 0, 1,
/// 2, or 3; following standard compiler practice.
static cl::opt<unsigned>
    OptLevel("cuabi-opt-level", cl::init(3), cl::NotHidden,
             cl::desc("Specify the GPU kernel optimization level."));
/// Enable expensive optimizations to allow the compiler to use the
/// maximum amount of resources (memory and time).  If the
/// optimization level if >= 2 this flag is enabled.
static cl::opt<bool>
    AllowExpensiveOpts("cuabi-enable-expensive-optimizations", cl::init(false),
                       cl::Hidden,
                       cl::desc("Enable expensive optimizations that use "
                                "maximum available resources."));
/// Disable the generatino of floating point multiply add instructions.
/// This is passed on to 'ptxas' to disable the contraction of floating
/// point multiply-add operations (FMAD, FFMA, or DFMA).  This is
/// equivalent to passing '-fmad false' to 'ptxas'.
static cl::opt<bool>
    DisableFMA("cuabi-disable-fma", cl::init(false), cl::Hidden,
               cl::desc("Disable the generation of FMA instructions."
                        "(default=false)"));
/// Disable the optimizer's constant bank optimizations.  Passed directly
/// to 'ptxas' as '--disable-optimizer-constants'.
static cl::opt<bool>
    DisableConstantBank("cuabi-disable-constant-bank", cl::init(false),
                        cl::Hidden,
                        cl::desc("Disable the use of the constants bank in "
                                 "GPU code generation. (default=false)"));

/// Set the CUDA ABI's default grainsize value.  This is used internally
/// by the transform.
static cl::opt<unsigned> DefaultGrainSize(
    "cuabi-default-grainsize", cl::init(1), cl::Hidden,
    cl::desc("The default grainsize used by the transform "
             "when analysis fails to determine one. (default=1)"));

/// Keep the complete set of intermediate files around after compilation.  This
/// includes LLVM IR, PTX, and the fatbinary file.
static cl::opt<bool> KeepIntermediateFiles(
    "cuabi-keep-files", cl::init(false), cl::Hidden,
    cl::desc("Keep all the intermediate files on disk after"
             "successsful completion of the transforms "
             "various steps."));

/// Generate code to prefetch data prior to kernel launches.  This is literally
/// in the few lines right before a launch so obviously less than ideal.
static cl::opt<bool>
    CodeGenDisablePrefetch("cuabi-disable-prefetch", cl::init(false),
                           cl::Hidden,
                           cl::desc("Disable insertion of calls to do data "
                                    "prefetching for UVM-based kernel  "
                                    "parameters."));

/// Should PTX files be included in the fat binary images produced by the
/// transform?
static cl::opt<bool>
    EmbedPTXInFatbinaries("cuabi-embed-ptx", cl::init(false),
                          cl::Hidden,
                          cl::desc("Embed intermediate PTX files in the "
                                   "fatbinaries used by the CUDA ABI "
                                   "transformation."));

/// Provide a hard-coded default value for the number of threads per block to
/// use in kernel launches.  This provides a compile-time mechanisms for
/// setting this value and it will persist throughout the execution of the
/// associated compilation unit(s).  The runtime internally currently uses the
/// equations,
///
///  ``unsigned blockSize = 4 * warpSize;``
///  ``blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;``
///
/// to determine the overall set of launch parameters.  This is mostly meant
/// for experimentation and testing.
static cl::opt<unsigned>
    DefaultThreadsPerBlock("cuabi-threads-per-block", cl::init(256), cl::Hidden,
                           cl::desc("Set the runtime system's value for "
                                    "the default number of threads per block. "
                                    "(default=256)"));

static cl::opt<unsigned>
    DefaultBlocksPerGrid("cuabi-blocks-per-grid", cl::init(0), cl::Hidden,
                         cl::desc("Hard-code the runtime system's value for "
                                  "the number of blocks per grid in kernel "
                                  "launches. (default=0=disabled)"));

/*
 * TODO: work here needs to be done to support these additional arguments
 * to expose more of the ptxas feature set within the ABI transform.
 *
 /// Specify the maximum number of registers that GPU functions can use.
 /// Until a function-specific limit, a higher value will generally
 /// increase the performance of individual GPU threads that execute this
 /// function. However, because thread registers are allocated from a
 /// global register pool on each GPU, a higher value of this option
 /// will also reduce the maximum thread block size, thereby reducing the
 /// amount of thread parallelism.
 static cl::opt<int>
 MaxRegCount("cuabi-maxregcount", cl::init(-1), cl::Hidden,
             cl::desc("Specify the max number t of registers that GPU functions
 " "can use."));
 /// Control the aggressiveness of optimizations that affect register usage.
 /// ([0..10], default = 5) Higher values aggressively optimize the source
 /// program, trading off additional register usage for potential
 /// improvements in the generated code. Lower values inhibit optimizations
 /// that aggressively increase register usage. This option can work in
 /// conjunction with -maxrregcount and kernel launch bounds. (NVIDIA states:
 /// This is a BETA feature for advanced users and there is no guarantee that
 /// the implementation stays consistent between ptxas releases.)
 static cl::opt<unsigned>
 RegUsageLevel("cuabi-register-usage-level", cl::init(5), cl::Hidden,
               cl::desc("Experimental feature -- see 'ptxas' documentaiton. "
                        "(default=5)"));
*/

// Adapted from Transforms/Utils/ModuleUtils.cpp
// TODO: Technically we only use this to add a global ctor for
// dealing with the nuances of CUDA kernels so perhaps we'd be
// off renaming this to match our specific use case...  ????
static void appendToGlobalArray(const char *Array, Module &M, Constant *C,
                                int Priority, Constant *Data) {

  IRBuilder<> IRB(M.getContext());
  FunctionType *FnTy = FunctionType::get(IRB.getVoidTy(), false);

  // Get the current set of static global constructors and add
  // the new ctor to the list.
  SmallVector<Constant *, 16> CurrentCtors;
  StructType *EltTy = StructType::get(
      IRB.getInt32Ty(), PointerType::getUnqual(FnTy), IRB.getInt8PtrTy());
  if (GlobalVariable *GVCtor = M.getNamedGlobal(Array)) {
    if (Constant *Init = GVCtor->getInitializer()) {
      unsigned N = Init->getNumOperands();
      CurrentCtors.reserve(N + 1);
      for (unsigned i = 0; i != N; ++i)
        CurrentCtors.push_back(cast<Constant>(Init->getOperand(i)));
    }
    GVCtor->eraseFromParent();
  }

  // Build a 3 field global_ctor entry.  We don't take a comdat key.
  Constant *CSVals[3];
  CSVals[0] = IRB.getInt32(Priority);
  CSVals[1] = C;
  CSVals[2] = Data ? ConstantExpr::getPointerCast(Data, IRB.getInt8PtrTy())
                   : Constant::getNullValue(IRB.getInt8PtrTy());
  Constant *RuntimeCtorInit =
      ConstantStruct::get(EltTy, makeArrayRef(CSVals, EltTy->getNumElements()));

  CurrentCtors.push_back(RuntimeCtorInit);

  // Create a new initializer.
  ArrayType *AT = ArrayType::get(EltTy, CurrentCtors.size());
  Constant *NewInit = ConstantArray::get(AT, CurrentCtors);

  // Create the new global variable and replace all uses of
  // the old global variable with the new one.
  (void)new GlobalVariable(M, NewInit->getType(), false,
                           GlobalValue::AppendingLinkage, NewInit, Array);
}

/// Take the NVIDIA CUDA 'sm_' architecture format and convert it into
/// the 'compute_' form.  Note that we require CUDA 11 or greater and
/// we have removed support for sm_2x and sm_3x architectures.

static std::string virtualArchForCudaArch(StringRef Arch) {
  // TODO: We've scaled back some from the full suite of Nvidia targets
  // as we are going in assuming we will support only CUDA 11 or greater.
  // We should probably raise an error for sm_2x and sm_3x -- this is
  // different than the current CUDA support in Clang and could be
  // confusing to users... That said, not sure it makes a lot of sense
  // to work on support for deprecated architectures (and thus older
  // versions of CUDA).
  return llvm::StringSwitch<std::string>(Arch)
      // sm_20 (Fermi) is deprecated as of CUDA 9.
      // sm_3X (Kepler) is deprecated as of CUDA 11.
      .Case("sm_50", "compute_50") // Maxwell (to be deprecated w/ CUDA 12?)
      .Case("sm_52", "compute_52") //
      .Case("sm_53", "compute_53") //
      .Case("sm_60", "compute_60") // Pascal
      .Case("sm_61", "compute_61") //
      .Case("sm_62", "compute_62") //
      .Case("sm_70", "compute_70") // Volta
      .Case("sm_72", "compute_72") //
      .Case("sm_75", "compute_75") // Turing
      .Case("sm_80", "compute_80") // Ampere
      // TODO: LLVM's PTX target doesn't appear to support
      // anything past sm_80???  For LLVM 12.x.  Need to
      // update for new architectures -- 13.x will support
      // through sm_86.
      .Case("sm_86", "compute_86") //
      .Case("sm_87", "compute_87") //
      .Default("unknown");
}

static std::string PTXVersionFromCudaVersion() {
#ifdef CUDATOOLKIT_VERSION
  Twine CudaVersionStr = Twine(CUDATOOLKIT_VERSION_MAJOR) + Twine(".") +
                         Twine(CUDATOOLKIT_VERSION_MINOR);
#else
#pragma message("warning, no CUDA Toolkit version info available.")
  Twine CudaVersionStr = "unknown";
#endif

  // TODO: There could a tighter connection here with the GPU
  // architecture choice that we are not cross referencing.
  return llvm::StringSwitch<std::string>(CudaVersionStr.str())
      // TODO: These CUDA to PTX version translations will have
      // to be watched between CUDA and LLVM resources.  It is
      // not uncommon for LLVM to lag behind CUDA PTX versions.
      // The details below are based on Cuda 11.6 and LLVM 13.x.
      .Case("10.0", "+ptx63")
      .Case("10.1", "+ptx64")
      .Case("10.2", "+ptx65")
      .Case("10.0", "+ptx63")
      .Case("11.0", "+ptx70")
      .Case("11.1", "+ptx71")
      .Case("11.2", "+ptx72")
      .Case("11.3", "+ptx72")
      .Case("11.4", "+ptx72")
      .Case("11.5", "+ptx72")
      .Case("11.6", "+ptx72")
      .Default("+ptx72"); // TODO: fall back or go with latest?
}

// Helper function to configure the details of our post-Tapir transformation
// passes.
//
// TODO: Need to spend some time exploring the selected set of passes here.
//
const unsigned OptLevel0 = 0;
const unsigned OptLevel1 = 1;
const unsigned OptLevel2 = 2;
const unsigned OptLevel3 = 3;

const unsigned SizeLevel0 = 0;
const unsigned SizeLevel1 = 1;
const unsigned SizeLevel2 = 2;

// NOTES: From the NVPTX target documentation.
//  (See: https://llvm.org/docs/NVPTXUsage.html)
//
//  The NVVMReflect pass should be executed early in the optimization
//  pipeline, immediately after the link stage. The internalize pass is also
//  recommended to remove unused math functions from the resulting PTX. For an
//  input IR module module.bc, the following compilation flow is recommended:
//
//  1. Save list of external functions in module.bc.
//  2. Link module.bc with libdevice.compute_XX.YY.bc.
//  3. Internalize all functions not in list from step #1.
//  4. Eliminate all unused internal functions.
//  5. Run NVVMReflect pass.
//  6. Run standard optimization pipeline.
//
//  The NVVMReflect pass replaces conditionals with constants and it will
//  often leave behind dead code. Therefore, it is recommended that
//  NVVMReflect be executed early in the optimization pipeline before
//  dead-code elimination.  The NVPTX TargetMachine knows how to schedule
//  NVVMReflect at the beginning of your pass manager; just use the following
//  code when setting up your pass manager:
//
//    std::unique_ptr<TargetMachine> TM = ...;
//    PassManagerBuilder PMBuilder(...);
//    if (TM)
//      TM->adjustPassManager(PMBuilder);
//

/// Add a set of customized optimization passes over the LLVM code
/// post Tapir transformation but prior to conversion to PTX.  The
/// 'OptLevel' parameter controls the compiler optimization level
/// for performance, while 'SizeLevel' controls the size optimization
/// target.
static void runOptimizationPasses(Module &KM, unsigned OptLevel = OptLevel3,
                                  unsigned SizeLevel = OptLevel2) {
  legacy::PassManager PM;
  PM.add(createReassociatePass());
  PM.add(createGVNPass());
  PM.add(createCFGSimplificationPass());
  PM.add(createSLPVectorizerPass());
  PM.add(createDeadCodeEliminationPass());
  PM.add(createDeadStoreEliminationPass());
  PM.add(createCFGSimplificationPass());
  PM.add(createDeadCodeEliminationPass());
  PM.add(createVerifierPass());
  PM.run(KM);
}

// NOTE: The NextKernelID variable below is not thread safe.
// This currently isn't an issue but if LLVM at some point
// in the future starts supporting multiple compilation threads
// (e.g. over Modules for example) the support code for kernel
// IDs will have a race...

/// Static ID for kernel naming -- each encountered kernel (loop)
/// during compilation will receive a unique ID.
unsigned CudaLoop::NextKernelID = 0;

CudaLoop::CudaLoop(Module &M, const std::string &KN,
                   CudaABI *TapirT, bool MakeUniqueName)
    // TODO: Is KernelModule used below before it is constructed?
    : LoopOutlineProcessor(M, KernelModule),
      KernelModule(KN.c_str() + Twine(NextKernelID).str(),
                   M.getContext()),
      TTarget(TapirT),
      KernelName(KN) {

  if (MakeUniqueName) {
    std::string UN = KN + "_" + Twine(NextKernelID).str();
    NextKernelID++; // rand();
    KernelName = UN;
  }

  LLVM_DEBUG(dbgs() << "cuabi: cuda loop outliner creation:\n"
                    << "\tbase kernel name: " << KernelName << "\n"
                    << "\tmodule name     : " << KernelModule.getName()
                    << "\n\n");

  PTXTargetMachine = nullptr;

  std::string ArchString;
  if (HostMArch == "64")
    ArchString = "nvptx64";
  else
    ArchString = "nvptx";

  // Note the "cuda" choice as part of the OS portion of the triple selects
  // compatability with the CUDA Driver API.
  Triple TT(ArchString, "nvidia", "cuda");

  std::string Error;
  const Target *PTXTarget = TargetRegistry::lookupTarget("", TT, Error);
  if (!PTXTarget) {
    errs() << "Target lookup failed: " << Error << "\n";
    report_fatal_error("Unable to find registered PTX target. "
                       "Was LLVM built with the NVPTX target enabled?");
  }

  // The feature string for the target machine can be confusing (this is the
  // 3rd parameter to createTargetMachine())).  The most common usage seems
  // to suggest a 32- or 64-bit mode (e.g., "+ptx64").  However, digging into
  // clang's implementation it turns out the feature string is actually a
  // PTX version specifier that goes along with CUDA version.
  std::string PTXVersionStr = PTXVersionFromCudaVersion();

  LLVM_DEBUG(dbgs() << "\tptx target feature version "
                    << PTXVersionStr
                    << ".\n");

  PTXTargetMachine = PTXTarget->createTargetMachine(
      TT.getTriple(), GPUArch, PTXVersionStr.c_str(), TargetOptions(),
      Reloc::PIC_, CodeModel::Small, CodeGenOpt::Aggressive);

  KernelModule.setTargetTriple(TT.str());
  // TODO: The data layout that this generates does not match that listed in
  // the NVPTX documentation: https://llvm.org/docs/NVPTXUsage.html#data-layout
  KernelModule.setDataLayout(PTXTargetMachine->createDataLayout());

  // Insert PTX intrinsic declarations into the host module.
  LLVMContext &Ctx = KernelModule.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  PointerType *Int8PtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *Int64PtrTy = Type::getInt64PtrTy(Ctx);
  PointerType *CharPtrTy = PointerType::getUnqual(Type::getInt8Ty(Ctx));


  GetThreadIdx = KernelModule.getOrInsertFunction("gtid", Int32Ty);

  // Thread index values -- equivalent to Cuda's builtins:  threadIdx.[x,y,z].
  CUThreadIdxX = Intrinsic::getDeclaration(&KernelModule,
                                           Intrinsic::nvvm_read_ptx_sreg_tid_x);
  CUThreadIdxY = Intrinsic::getDeclaration(&KernelModule,
                                           Intrinsic::nvvm_read_ptx_sreg_tid_y);
  CUThreadIdxZ = Intrinsic::getDeclaration(&KernelModule,
                                           Intrinsic::nvvm_read_ptx_sreg_tid_z);

  // Block index values -- equivalent to Cuda's builtins: blockIndx.[x,y,z].
  CUBlockIdxX = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
  CUBlockIdxY = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ctaid_y);
  CUBlockIdxZ = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ctaid_z);

  // Block dimensions -- equivalent to Cuda's builtins: blockDim.[x,y,z].
  CUBlockDimX = Intrinsic::getDeclaration(&KernelModule,
                                          Intrinsic::nvvm_read_ptx_sreg_ntid_x);
  CUBlockDimY = Intrinsic::getDeclaration(&KernelModule,
                                          Intrinsic::nvvm_read_ptx_sreg_ntid_y);
  CUBlockDimZ = Intrinsic::getDeclaration(&KernelModule,
                                          Intrinsic::nvvm_read_ptx_sreg_ntid_x);

  // Grid dimensions -- equivalent to Cuda's builtins: gridDim.[x,y,z].
  CUGridDimX = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_nctaid_x);
  CUGridDimY = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_nctaid_y);
  CUGridDimZ = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_nctaid_z);

  // NVVM-centric barrier -- equivalent to Cuda's __sync_threads().
  CUSyncThreads =
      Intrinsic::getDeclaration(&KernelModule, Intrinsic::nvvm_barrier0);

  // Get entry points into the Cuda-centric portion of the Kitsune GPU runtime.
  // These are a layer deeper than the interface used by the GPUABI.  While we
  // could codegen straight to the Cuda (Driver) API, the higher level calls
  // help to simplify codegen calls.
  KitCudaInitFn = M.getOrInsertFunction("__kitrt_cuInit", VoidTy);
  KitCudaCtxCheckFn = M.getOrInsertFunction("__kitrt_cuCheckCtxState", VoidTy);
  KitCudaLaunchFn = M.getOrInsertFunction("__kitrt_cuLaunchFBKernel",
                                          VoidPtrTy, // returns an opaque stream
                                          VoidPtrTy, // fatbinary
                                          VoidPtrTy, // kernel name
                                          VoidPtrPtrTy, // arguments
                                          Int64Ty);     // trip count
  KitCudaLaunchModuleFn =
                    M.getOrInsertFunction("__kitrt_cuLaunchModuleKernel",
                                          VoidPtrTy, // returns opaque stream
                                          VoidPtrTy, // CUDA module
                                          VoidPtrTy, // kernel name
                                          VoidPtrPtrTy, // arguments
                                          Int64Ty); // trip count
  KitCudaWaitFn =
      M.getOrInsertFunction("__kitrt_cuStreamSynchronize", VoidTy, VoidPtrTy);
  KitCudaMemPrefetchFn =
      M.getOrInsertFunction("__kitrt_cuMemPrefetch", VoidTy, VoidPtrTy);
  KitCudaSetDefaultTBPFn =
      M.getOrInsertFunction("__kitrt_cuSetDefaultThreadsPerBlock", VoidTy,
                            Int32Ty); // threads per block.
  KitCudaSetDefaultLaunchParamsFn =
      M.getOrInsertFunction("__kitrt_cuSetDefaultLaunchParameters", VoidTy,
                            Int32Ty,  // blocks per grid
                            Int32Ty); // threads per block
  KitCudaCreateFBModuleFn =
      M.getOrInsertFunction("__kitrt_cuCreateFBModule", VoidPtrTy, VoidPtrTy);
  KitCudaGetGlobalSymbolFn =
      M.getOrInsertFunction("__kitrt_cuGetGlobalSymbol",
                            VoidPtrTy,  // return the device pointer for symbol.
                            CharPtrTy,  // symbol name
                            VoidPtrTy); // CUDA module

  KitCudaMemcpySymbolToDeviceFn =
      M.getOrInsertFunction("__kitrt_cuMemcpySymbolToDevice",
                            VoidTy,    // returns
                            VoidPtrTy, // host pointer
                            VoidPtrTy, // device pointer
                            Int64Ty);  // number of bytes to copy
}

CudaLoop::~CudaLoop() {}

// TODO: This call assumes we want to create the constant string
// in a fixed module ('M' in this case).  Perhaps should consider
// passing a Module to make things a bit more flexible?
Constant *CudaLoop::createConstantStr(const std::string &Str,
                                      const std::string &Name,
                                      const std::string &SectionName,
                                      unsigned Alignment) {
  LLVMContext &Ctx = M.getContext();
  Type *SizeTy = Type::getInt64Ty(Ctx);

  Constant *CSN = ConstantDataArray::getString(Ctx, Str);
  GlobalVariable *GV = new GlobalVariable(
      M, CSN->getType(), true, GlobalVariable::PrivateLinkage, CSN, ".devstr");
  Type *StrTy = GV->getType();

  const DataLayout &DL = M.getDataLayout();
  Constant *Zeros[] = {ConstantInt::get(DL.getIndexType(StrTy), 0),
                       ConstantInt::get(DL.getIndexType(StrTy), 0)};
  if (!SectionName.empty()) {
    GV->setSection(SectionName);
    // Mark the address as used which make sure that this section isn't
    // merged and we will really have it in the object file.
    GV->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::None);
  }

  if (Alignment)
    GV->setAlignment(llvm::Align(Alignment));

  Constant *CS = ConstantExpr::getGetElementPtr(GV->getValueType(), GV, Zeros);
  return CS;
}

void CudaLoop::setupLoopOutlineArgs(Function &F, ValueSet &HelperArgs,
                                    SmallVectorImpl<Value *> &HelperInputs,
                                    ValueSet &InputSet,
                                    const SmallVectorImpl<Value *> &LCArgs,
                                    const SmallVectorImpl<Value *> &LCInputs,
                                    const ValueSet &TLInputsFixed) {

  // Add the loop control inputs -- the first parameter defines
  // the extent of the index space (the number of threads to launch).
  {
    Argument *EndArg = cast<Argument>(LCArgs[1]);
    EndArg->setName("runSize");
    HelperArgs.insert(EndArg);

    Value *InputVal = LCInputs[1];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // The second parameter defines the start of the
  // index space.
  {
    Argument *StartArg = cast<Argument>(LCArgs[0]);
    StartArg->setName("runStart");
    HelperArgs.insert(StartArg);

    Value *InputVal = LCInputs[0];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // The third parameter defines the grainsize, if it is
  // not constant.
  if (!isa<ConstantInt>(LCInputs[2])) {
    Argument *GrainsizeArg = cast<Argument>(LCArgs[2]);
    GrainsizeArg->setName("grainSize");
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

void CudaLoop::updateKernelName(const std::string &KN, bool addID) {
  // Use the kernel ID to create a unique name.
  if (addID) {
    std::string UN = KN + "_" + Twine(NextKernelID).str();
    NextKernelID = rand();
    KernelName = UN;
  } else
    KernelName = KN;
}

static std::set<GlobalValue *> &collect(Constant &c,
                                        std::set<GlobalValue *> &seen);

static std::set<GlobalValue *> &collect(BasicBlock &bb,
                                        std::set<GlobalValue *> &seen) {
  for (auto &inst : bb)
    for (auto &op : inst.operands())
      if (auto *c = dyn_cast<Constant>(&op))
        collect(*c, seen);
  return seen;
}

static std::set<GlobalValue *> &collect(Function &f,
                                        std::set<GlobalValue *> &seen) {
  llvm::errs() << "collect function: " << f.getName() << "\n";
  seen.insert(&f);

  for (auto &bb : f)
    collect(bb, seen);
  return seen;
}

static std::set<GlobalValue *> &collect(GlobalVariable &g,
                                        std::set<GlobalValue *> &seen) {
  seen.insert(&g);

  if (g.hasInitializer())
    collect(*g.getInitializer(), seen);
  return seen;
}

static std::set<GlobalValue *> &collect(GlobalIFunc &g,
                                        std::set<GlobalValue *> &seen) {
  seen.insert(&g);

  llvm_unreachable("kitsune: GNU IFUNC not yet supported");
  return seen;
}

static std::set<GlobalValue *> &collect(GlobalAlias &g,
                                        std::set<GlobalValue *> &seen) {
  seen.insert(&g);

  llvm_unreachable("kitsune: GlobalAlias not yet supported");
  return seen;
}

static std::set<GlobalValue *> &collect(BlockAddress &blkaddr,
                                        std::set<GlobalValue *> &seen) {
  if (Function *f = blkaddr.getFunction())
    collect(*f, seen);
  if (BasicBlock *bb = blkaddr.getBasicBlock())
    collect(*bb, seen);
  return seen;
}

std::set<GlobalValue *> &collect(Constant &c, std::set<GlobalValue *> &seen) {
  if (GlobalValue *g = dyn_cast<GlobalValue>(&c))
    if (seen.find(g) != seen.end())
      return seen;

  if (auto *f = dyn_cast<Function>(&c))
    return collect(*f, seen);
  else if (auto *g = dyn_cast<GlobalVariable>(&c))
    return collect(*g, seen);
  else if (auto *g = dyn_cast<GlobalAlias>(&c))
    return collect(*g, seen);
  else if (auto *g = dyn_cast<GlobalIFunc>(&c))
    return collect(*g, seen);
  else if (auto *blkaddr = dyn_cast<BlockAddress>(&c))
    return collect(*blkaddr, seen);
  else
    for (auto &op : c.operands())
      if (auto *cop = dyn_cast<Constant>(op))
        collect(*cop, seen);
  return seen;
}

void CudaLoop::preProcessTapirLoop(TapirLoopInfo &TL, ValueToValueMapTy &VMap) {

  // TODO: process loop prior to outlining to do GPU/CUDA-specific things
  // like capturing global variables, etc.
  LLVM_DEBUG(dbgs() << "\tpreprocessing tapir loop...\n");

  // Collect the top-level entities (Function, GlobalVariable, GlobalAlias
  // and GlobalIFunc) that are used in the outlined loop. Since the outlined
  // loop will live in the KernelModule, any GlobalValue's used in it will
  // need to be cloned into the KernelModule and then register with CUDA
  // in the CUDA-centric ctor.
  std::set<GlobalValue *> UsedGlobalValues;

  Loop &L = *TL.getLoop();

  for (Loop *SL : L)
    for (BasicBlock *BB : SL->blocks())
      collect(*BB, UsedGlobalValues);

  for (BasicBlock *BB : L.blocks())
    collect(*BB, UsedGlobalValues);

  // Clone global variables (TODO: and aliases).
  for (GlobalValue *V : UsedGlobalValues) {
    if (GlobalVariable *G = dyn_cast<GlobalVariable>(V)) {
      // TODO: Make sure this logic makes sense...
      //
      // We don't necessarily need a GPU-side clone of a
      // global variable -- instead we need a location where
      // we can copy symbol information over from the host.
      GlobalVariable *NewG = new GlobalVariable(
          KernelModule, G->getValueType(), false,
          GlobalValue::ExternalWeakLinkage,
          (Constant *) Constant::getNullValue(G->getValueType()),
          G->getName() + "_devvar",
          (GlobalVariable *)nullptr);
      VMap[G] = NewG;
      LLVM_DEBUG(dbgs() << "\tcreated kernel-side global '"
                        << NewG->getName() << "'.\n"
                        << "\t\tsaving pair (" << G->getName()
                        << ", " << NewG->getName() << ")\n");
      GVarList.push_back(G);
    } else if (GlobalAlias *A = dyn_cast<GlobalAlias>(V)) {
      llvm_unreachable("kitsune: GlobalAlias not implemented.");
    }
  }

  // Create declarations for all functions first. These may be needed in the
  // global variables and aliases.
  for (GlobalValue *G : UsedGlobalValues) {
    if (Function *F = dyn_cast<Function>(G)) {
      Function *DeviceF = KernelModule.getFunction(F->getName());
      if (not DeviceF)
        DeviceF = Function::Create(F->getFunctionType(), F->getLinkage(),
                                   F->getName(), KernelModule);
      for (auto i = 0; i < F->arg_size(); i++) {
        Argument *Arg = F->getArg(i);
        Argument *NewA = DeviceF->getArg(i);
        NewA->setName(Arg->getName());
        VMap[Arg] = NewA;
      }
      VMap[F] = DeviceF;
    }
  }

  // FIXME: Support GlobalIFunc at some point. This is a GNU extension, so we
  // may not want to support it at all, but just in case, this is here.
  for (GlobalValue *V : UsedGlobalValues) {
    if (GlobalIFunc *G = dyn_cast<GlobalIFunc>(V)) {
      llvm_unreachable("kitsune: GlobalIFunc not yet supported.");
    }
  }

  // Now clone any function bodies that need to be cloned. This should be
  // done as late as possible so that the VMap is populated with any other
  // global values that need to be remapped.
  for (GlobalValue *v : UsedGlobalValues) {
    if (Function *F = dyn_cast<Function>(v)) {
      if (F->size()) {
        SmallVector<ReturnInst *, 8> Returns;
        Function *DeviceF = cast<Function>(VMap[F]);
        CloneFunctionInto(DeviceF, F, VMap,
                          CloneFunctionChangeType::DifferentModule, Returns);
        // GPU calls are slow, try to force inlining...
        DeviceF->addFnAttr(Attribute::AlwaysInline);
      }
    }
  }

  // Create declarations for all functions first. These may be needed in the
  // global variables and aliases.
  for (GlobalValue *G : UsedGlobalValues) {
    if (Function *F = dyn_cast<Function>(G)) {
      Function *DeviceF = KernelModule.getFunction(F->getName());
      if (not DeviceF)
        DeviceF = Function::Create(F->getFunctionType(), F->getLinkage(),
                                   F->getName(), KernelModule);
      for (auto i = 0; i < F->arg_size(); i++) {
        Argument *Arg = F->getArg(i);
        Argument *NewA = DeviceF->getArg(i);
        NewA->setName(Arg->getName());
        VMap[Arg] = NewA;
      }
      VMap[F] = DeviceF;
    }
  }
}

void CudaLoop::postProcessOutline(TapirLoopInfo &TLI, TaskOutlineInfo &Out,
                                  ValueToValueMapTy &VMap) {
  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Task *T = TLI.getTask();
  Loop *TL = TLI.getLoop();

  BasicBlock *Entry = cast<BasicBlock>(VMap[TL->getLoopPreheader()]);
  BasicBlock *Header = cast<BasicBlock>(VMap[TL->getHeader()]);
  BasicBlock *Exit = cast<BasicBlock>(VMap[TLI.getExitBlock()]);
  PHINode *PrimaryIV = cast<PHINode>(VMap[TLI.getPrimaryInduction().first]);
  Value *PrimaryIVInput = PrimaryIV->getIncomingValueForBlock(Entry);

  Instruction *ClonedSyncReg =
      cast<Instruction>(VMap[T->getDetach()->getSyncRegion()]);

  // We no longer need the cloned sync region.
  ClonedSyncReg->eraseFromParent();

  // Set the helper function to have external linkage.
  Function *Helper = Out.Outline;
  Helper->setName(KernelName);
  Helper->setLinkage(Function::ExternalLinkage);
  // Set the target features for the helper.
  AttrBuilder Attrs;
  Attrs.addAttribute("target-cpu", GPUArch);
  Attrs.addAttribute("target-features",
                     PTXVersionFromCudaVersion() + "," + GPUArch);
  Helper->removeFnAttr("target-cpu");
  Helper->removeFnAttr("target-features");
  Helper->removeFnAttr("personality");
  Helper->addAttributes(AttributeList::FunctionIndex, Attrs);

  NamedMDNode *Annotations =
      KernelModule.getOrInsertNamedMetadata("nvvm.annotations");
  SmallVector<Metadata *, 3> AV;
  AV.push_back(ValueAsMetadata::get(Helper));
  AV.push_back(MDString::get(Ctx, "kernel"));
  AV.push_back(
      ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 1)));
  Annotations->addOperand(MDNode::get(Ctx, AV));

  // Verify that the Thread ID corresponds to a valid iteration.  Because
  // Tapir loops use canonical induction variables, valid iterations range
  // from 0 to the loop limit with stride 1.  The End argument encodes the
  // loop limit. Get end and grainsize arguments
  Argument *End;
  Value *Grainsize;
  {
    auto OutlineArgsIter = Helper->arg_begin();
    // End argument is the first LC arg.
    End = &*OutlineArgsIter++;

    // Get the grainsize value, which is either constant or the third LC
    // arg.
    if (unsigned ConstGrainsize = TLI.getGrainsize())
      Grainsize = ConstantInt::get(PrimaryIV->getType(), ConstGrainsize);
    else
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
  Value *ThreadID = B.CreateIntCast(
      B.CreateAdd(ThreadIdx, B.CreateMul(BlockIdx, BlockDim), "thread_id"),
      PrimaryIV->getType(), false);
  ThreadID = B.CreateMul(ThreadID, Grainsize);
  Value *ThreadEnd = B.CreateAdd(ThreadID, Grainsize);
  Value *Cond = B.CreateICmpUGE(ThreadID, End);
  ReplaceInstWithInst(Entry->getTerminator(),
                      BranchInst::Create(Exit, Header, Cond));
  // Use the thread ID as the start iteration number for the primary IV.
  PrimaryIVInput->replaceAllUsesWith(ThreadID);

  // Update cloned loop condition to use the thread-end value.
  unsigned TripCountIdx = 0;
  ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TLI.getCondition()]);
  if (ClonedCond->getOperand(0) != ThreadEnd)
    ++TripCountIdx;
  ClonedCond->setOperand(TripCountIdx, ThreadEnd);
  assert(ClonedCond->getOperand(TripCountIdx) == ThreadEnd &&
         "End argument not used in condition");
}

void CudaLoop::transformForPTX() {
  LLVMContext &Ctx = KernelModule.getContext();
  Function &F = *KernelModule.getFunction(KernelName.c_str());

  // ThreadID.x
  auto tid = Intrinsic::getDeclaration(&KernelModule,
                                       Intrinsic::nvvm_read_ptx_sreg_tid_x);

  // BlockID.x
  auto ctaid = Intrinsic::getDeclaration(&KernelModule,
                                         Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
  // BlockDim.x
  auto ntid = Intrinsic::getDeclaration(&KernelModule,
                                        Intrinsic::nvvm_read_ptx_sreg_ntid_x);

  IRBuilder<> B(F.getEntryBlock().getFirstNonPHI());

  // Compute blockDim.x * blockIdx.x + threadIdx.x;
  Value *tidv = B.CreateCall(tid, {}, "thread_idx");
  Value *ntidv = B.CreateCall(ntid, {}, "block_idx");
  Value *ctaidv = B.CreateCall(ctaid, {}, "block_dimx");
  Value *tidoff = B.CreateMul(ctaidv, ntidv, "block_off");
  Value *gtid = B.CreateAdd(tidoff, tidv, "cu_idx");

  // PTX doesn't like .<n> global names, rename them to
  // replace the '.' with an underscore, '_'.
  for (GlobalVariable &G : KernelModule.globals()) {
    auto name = G.getName().str();
    std::replace(name.begin(), name.end(), '.', '_');
    G.setName(name);
  }

  // Check if there are unresolved sumbbols to see if we might need
  // libdevice
  std::set<std::string> Unresolved;
  for (auto &F : KernelModule) {
    if (F.hasExternalLinkage())
      Unresolved.insert(F.getName().str());
  }

  if (!Unresolved.empty()) {
    // Load libdevice and check for provided functions
    llvm::SMDiagnostic SMD;
    Optional<std::string> Path = sys::Process::FindInEnvPath(
        "CUDA_HOME", "nvvm/libdevice/libdevice.10.bc");

    if (!Path) {
      report_fatal_error("Cuda ABI transform: failed to find libdevice!");
    }

    std::unique_ptr<llvm::Module> LibDevice = parseIRFile(*Path, SMD, Ctx);
    if (!LibDevice)
      report_fatal_error("cuda abi transform: failed to parse libdevice!");

    // We iterate through the provided functions of the moodule and if there
    // are remaining function calls we add them.
    std::set<std::string> Provided;
    std::string NVPref = "__nv_";
    for (auto &F : *LibDevice) {
      std::string Name = F.getName().str();
      auto Res = std::mismatch(NVPref.begin(), NVPref.end(), Name.begin());
      auto OldName = Name.substr(Res.second - Name.begin());
      if (Res.first == NVPref.end() && Unresolved.count(OldName) > 0)
        Provided.insert(OldName);
    }
    for (auto &Fn : Provided) {
      if (auto *F = KernelModule.getFunction(Fn))
        F->setName(NVPref + Fn);
    }

    auto L = Linker(KernelModule);
    L.linkInModule(std::move(LibDevice), 2);
  }

  std::vector<Instruction *> TIDs;
  for (auto &F : KernelModule) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (Function *F = CI->getCalledFunction()) {
            if (F->getName() == "gtid")
              TIDs.push_back(&I);
          }
        }
      }
    }
  }

  for (auto P : TIDs) {
    P->replaceAllUsesWith(gtid);
    P->eraseFromParent();
  }

  if (auto *F = KernelModule.getFunction("gtid"))
    F->eraseFromParent();
}

std::string CudaLoop::createPTXFile() {

  assert(PTXTargetMachine && "can't emit PTX w/out machine target");

  // There were a couple of paths here to consider for generating PTX
  // so here's a high-level summary.  The first option is to directly
  // use the Cuda API within the transform (like the JIT-based runtime
  // target does).  The second approach is to use a Cuda toolchain
  // approach via the use of intermediate files produced by the
  // transform.
  //
  // The most significant drawbacks of the first approach are a heavy
  // change in the LLVM CMake config to include all the bells-and-whistles
  // for adding Cuda into the mix.  In addition, use of the API would
  // require an appropriate GPU in the system where the compiler is being
  // run (eliminating/hindering some cross-compilation use cases).  An
  // advantage of the approach is that it eliminates the creation of
  // external files and (perhaps) slows down overall compilation times.
  //
  // Given a heavily used cross-compilation approach in a number of use
  // cases this transform uses the non-API implementation.  This also
  // closely matches the implementation approach used for Cuda by Clang.
  // While it isn't easy to share that code here, it does provide us with
  // a reference implementation.
  std::error_code EC;
  std::string PTXFileName = KernelName + ".ptx";
  std::unique_ptr<ToolOutputFile> PTXFile;
  PTXFile = std::make_unique<ToolOutputFile>(PTXFileName, EC,
                                             sys::fs::OpenFlags::OF_None);

  // TODO: Update when we merge with LLVM >= 13.x to use the
  // PTXFile->outputFilename() so we get the fully qualified
  // path here...
  LLVM_DEBUG(dbgs() << "\tgenerating ptx file: " << PTXFile->getFilename()
                    << ".\n");

  LLVMContext &Ctx = KernelModule.getContext();
  Function &F = *KernelModule.getFunction(KernelName.c_str());

  KernelModule.addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz", true);

  legacy::PassManager PM;
  legacy::FunctionPassManager FPM(&KernelModule);
  PassManagerBuilder Builder;
  Builder.OptLevel = OptLevel;
  Builder.VerifyInput = 1;
  Builder.Inliner = createFunctionInliningPass(Builder.OptLevel, 0, false);
  Builder.populateLTOPassManager(PM);
  Builder.populateFunctionPassManager(FPM);
  Builder.populateModulePassManager(PM);

  bool Fail;
  Fail = PTXTargetMachine->addPassesToEmitFile(
      PM, PTXFile->os(), nullptr, CodeGenFileType::CGFT_AssemblyFile, false);
  if (Fail)
    report_fatal_error("An unknown error caused the Tapir Cuda-abi "
                       "transform to fail, PTX code could not be emitted.");

  // Run the configured optimization passes and save the PTX file.
  FPM.doInitialization();
  if (PTXTargetMachine)
    PTXTargetMachine->adjustPassManager(Builder);
  for (Function &Fn : KernelModule)
    FPM.run(Fn);
  FPM.doFinalization();
  PM.run(KernelModule);

  // Don't automatically clean up the PTX file -- the next step will be to
  // convert it into a gpu-architecture-target s-code via the 'ptxas'
  // component in the Cuda toolchain...
  PTXFile->keep();
  TTarget->addToPTXFileList(PTXFile->getFilename().str());
  return PTXFile->getFilename().str();
}

std::string CudaLoop::createFatBinaryFile(const std::string &PTXFileName) {

  LLVM_DEBUG(dbgs() << "creating fat binary file from ptx source file.\n");

  std::error_code EC;
  std::string AsmFileName = KernelName + ".s";
  std::unique_ptr<ToolOutputFile> AsmFile;
  AsmFile = std::make_unique<ToolOutputFile>(AsmFileName, EC,
                                             sys::fs::OpenFlags::OF_None);

  std::string FBFileName = KernelName + ".fatbin";
  std::unique_ptr<ToolOutputFile> FBFile;
  FBFile = std::make_unique<ToolOutputFile>(FBFileName, EC,
                                            sys::fs::OpenFlags::OF_None);
  // TODO: LLVM docs suggest we shouldn't be using findProgramByName()
  // as a fully qualified path is preferred -- punting on this for now.
  auto PTXASExe = sys::findProgramByName("ptxas");
  if ((EC = PTXASExe.getError()))
    report_fatal_error("'ptxas' not found. "
                       "Is a CUDA installation in your path?");
  LLVM_DEBUG(dbgs() << "ptxas: " << *PTXASExe << "\n");

  opt::ArgStringList PTXASArgList;
  opt::ArgStringList FatBinArgList;

  PTXASArgList.push_back(PTXASExe->c_str());
  PTXASArgList.push_back("-c");
  PTXASArgList.push_back("--gpu-name");
  PTXASArgList.push_back(GPUArch.c_str());
  PTXASArgList.push_back("--machine");
  PTXASArgList.push_back(HostMArch.c_str());

  if (Verbose)
    PTXASArgList.push_back("--verbose");

  if (Debug)
    PTXASArgList.push_back("--device-debug");
  if (GenLineInfo)
    PTXASArgList.push_back("--generate-line-info");
  if (BoundsCheck) {
    if (OptLevel != 0 || !Debug)
      PTXASArgList.push_back("--sp-bounds-check");
  }

  if (SurpressDBInfo) {
    if (Debug || GenLineInfo)
      PTXASArgList.push_back("--suppress-debug-info");
    else
      errs() << "warning: ignoring -tapir-cu-surpress-debug-info as it "
             << "requires debug or generate-line-info option.\n";
  }

  if (OptLevel > 3) {
    errs() << "warning: -tapir-cu-opt-level=" << OptLevel
           << " is not supported, using level 3 instead.\n";
    OptLevel = 3;
  }
  PTXASArgList.push_back("--opt-level");
  switch (OptLevel) {
  case 0:
    PTXASArgList.push_back("0");
    break;
  case 1:
    PTXASArgList.push_back("1");
    break;
  case 2:
    PTXASArgList.push_back("2");
    break;
  case 3:
    PTXASArgList.push_back("3");
    break;
  default:
    llvm_unreachable_internal("unhandled optimization level", __FILE__,
                              __LINE__);
    break;
  }

  if (AllowExpensiveOpts && OptLevel < 2) {
    PTXASArgList.push_back("--allow-expensive-optimizations");
    PTXASArgList.push_back("true");
  }

  if (DisableFMA) {
    PTXASArgList.push_back("--fmad");
    PTXASArgList.push_back("false");
  }

  if (DisableConstantBank)
    PTXASArgList.push_back("--disable-optimizer-constants");

  if (KernelModule.getNamedMetadata("llvm.dbg.cu")) {
    PTXASArgList.push_back("-g");
    PTXASArgList.push_back("--generate-line-info");
  }

  LLVM_DEBUG(dbgs() << "\tcreating s-code file: "
                    << AsmFile->getFilename().str(););
  PTXASArgList.push_back("--output-file");
  std::string SCodeFilename = AsmFile->getFilename().str();
  PTXASArgList.push_back(SCodeFilename.c_str());
  PTXASArgList.push_back(PTXFileName.c_str());
  PTXASArgList.push_back(nullptr);

  // Run ptxas to convert PTX to a s-code/GPU assembly file.
  SmallVector<const char *, 128> PTXASArgv;
  PTXASArgv.append(PTXASArgList.begin(), PTXASArgList.end());
  PTXASArgv.push_back(nullptr);
  auto PTXASArgs = toStringRefArray(PTXASArgv.data());

  LLVM_DEBUG(dbgs() << "\tptxas command line:\n"; unsigned c = 0;
             for (auto dbg_arg
                  : PTXASArgs) {
               dbgs() << "\t\t" << c << ": " << dbg_arg << "\n";
               c++;
             });

  std::string ErrMsg;
  bool ExecFailed;
  int ExecStat = sys::ExecuteAndWait(*PTXASExe, PTXASArgs, None, {},
                                     0, /* secs to wait -- 0 --> unlimited */
                                     0, /* memory limit -- 0 --> unlimited */
                                     &ErrMsg, &ExecFailed);
  if (ExecFailed)
    report_fatal_error("unable to execute 'ptxas'.");

  if (ExecStat != 0)
    // 'ptxas' ran but returned an error state.
    // TODO: Need to check what sort of actual state 'ptxas'
    // returns to the environment -- currently assuming it
    // matches standard practices...
    report_fatal_error("ptxas execution error:" + ErrMsg);

  AsmFile->keep();
  FBFile->keep();

  // TODO: LLVM docs suggest we shouldn't be using findProgramByName()...
  auto FatBinExe = sys::findProgramByName("fatbinary");
  if (EC = FatBinExe.getError())
    report_fatal_error("'fatbinary' not found. "
                       "Is a CUDA installation in your path?");
  LLVM_DEBUG(dbgs() << "\tfatbinary: " << *FatBinExe << "\n");

  FatBinArgList.push_back(FatBinExe->c_str());

  //FatBinArgList.push_back("-c"); // make relocatable

  if (HostMArch == "32")
    FatBinArgList.push_back("--32");
  else if (HostMArch == "64")
    FatBinArgList.push_back("--64");

  FatBinArgList.push_back("--create");
  std::string FatbinFilename = FBFile->getFilename().str();
  FatBinArgList.push_back(FatbinFilename.c_str());
  std::string FBImgArgs =
      std::string("--image=profile=") + GPUArch + ",file=" + SCodeFilename;
  FatBinArgList.push_back(FBImgArgs.c_str());
  std::string VArch = virtualArchForCudaArch(GPUArch);
  if (VArch == "unknown") {
    std::string Msg = GPUArch + ": unsupported CUDA architecture!";
    report_fatal_error(Msg.c_str());
  }
  std::string PTXImgArgs =
      std::string("--image=profile=") + VArch + ",file=" + PTXFileName.c_str();
  FatBinArgList.push_back(PTXImgArgs.c_str());
  FatBinArgList.push_back(nullptr);

  SmallVector<const char *, 128> FatBinArgv;
  FatBinArgv.append(FatBinArgList.begin(), FatBinArgList.end());
  auto FatBinArgs = toStringRefArray(FatBinArgv.data());

  LLVM_DEBUG({
    dbgs() << "\tfatbinary command line:\n";
    unsigned c = 0;
    for (auto dbg_arg : FatBinArgs) {
      dbgs() << "\t\t" << c << ": " << dbg_arg << "\n";
      c++;
    }
  });

  ExecStat = sys::ExecuteAndWait(*FatBinExe, FatBinArgs, None, {},
                                 0, /* secs to wait -- 0 --> unlimited */
                                 0, /* memory limit -- 0 --> unlimited */
                                 &ErrMsg, &ExecFailed);

  if (ExecFailed)
    report_fatal_error("unable to execute 'fatbinary'.");

  if (ExecStat != 0)
    // 'fatbinary' ran but returned an error state.
    // TODO: Need to check what sort of actual state 'fatbinary'
    // returns to the environment -- currently assuming it
    // matches standard practices...
    report_fatal_error("'fatbinary' error:" + ErrMsg);

  FBFile->keep();

  // Take the next phase of steps required to embed the fat binary
  // into the generated code.
  return std::string(FBFile->getFilename().str());
}

// Create a global variable that contains the contents of a fat binary file
// created using the cuda 'ptxas' and 'fatbinary' components from the cuda
// toolchain.
Constant *CudaLoop::createKernelBuffer() {

  // Before we move along too far check to see if want to save the kernel
  // module -- this can be helpful if we bomb-out in PTX code generation
  // below.
  if (KeepIntermediateFiles) {
    std::error_code EC;
    std::unique_ptr<ToolOutputFile> KMFile;
    std::string KMFileName = KernelName + ".ll";
    KMFile = std::make_unique<ToolOutputFile>(KMFileName, EC,
                                              sys::fs::OpenFlags::OF_None);
    if (EC) {
      errs() << "failed to create kernel module IR output file '" << KMFileName
             << "'.\n"
             << EC.message() << ".\n";
    } else {
      KernelModule.print(KMFile->os(), nullptr);
      KMFile->keep();
    }
  }

  // In order to support cross-compilation use cases we use an approach very
  // similar to clang for cuda support -- we transform the loop (kernel)
  // into ptx file, use 'ptxas' to assemble it and then take the result and
  // use 'fatbinary' to create the binary kernel that will be embedded into
  // the executable.
  std::string PTXFileName = createPTXFile();
  if (PTXFileName.empty()) {
    report_fatal_error("PTX file generation failed!");
    // we'll likely never make it here given the fatal error call above...
    return nullptr;
  }

  LLVM_DEBUG(dbgs() << "\tfatbinary stage: ptx file '" << PTXFileName
                    << "' created.\n");

  std::string FBFileName = createFatBinaryFile(PTXFileName);
  if (FBFileName.empty()) {
    report_fatal_error("Fatbinary file generation failed!");
    // we'll likely never make it here given the fatal error call above...
    return nullptr;
  }

  LLVM_DEBUG(dbgs() << "\tfatbinary stage: fatbin file '" << FBFileName
                    << "' created.\n");

  // We now have a fatbinary image file on disk.  Start the process of
  // embedding it into the executable by creating a global variable to
  // hold the contents of the fatbin file.  The resulting variable is
  // our return value...
  std::unique_ptr<llvm::MemoryBuffer> FBBuffer = nullptr;
  ErrorOr<std::unique_ptr<MemoryBuffer>> FBBufferOrErr =
      MemoryBuffer::getFile(FBFileName);
  if (std::error_code EC = FBBufferOrErr.getError()) {
    report_fatal_error("failed to load cuda fat binary image: " + EC.message());
    return nullptr; // no-op.
  }

  FBBuffer = std::move(FBBufferOrErr.get());
  LLVM_DEBUG(dbgs() << "\tloaded fat binary file size is "
                    << FBBuffer->getBufferSize() << " bytes.\n");
  LLVMContext &Ctx = M.getContext();

  const char *FatbinConstantName = ".nv_fatbin";
  const char *FatbinSectionName = ".nvFatBinSegment";
  const char *ModuleSectionName = "__nv_module_id";

  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Constant *FBCS = ConstantDataArray::getRaw(
      StringRef(FBBuffer->getBufferStart(), FBBuffer->getBufferSize()),
      FBBuffer->getBufferSize(), Int8Ty);
  GlobalVariable *GV;
  GV = new GlobalVariable(M, FBCS->getType(), true, GlobalValue::PrivateLinkage,
                          FBCS, getKernelName() + Twine("_fatbin"));

  LLVM_DEBUG(dbgs() << "\tcreated fat binary global variable '" << GV->getName()
                    << "'.\n");

  Type *StrTy = GV->getType();
  const DataLayout &DL = M.getDataLayout();
  Constant *Zeros[] = {ConstantInt::get(DL.getIndexType(StrTy), 0),
                       ConstantInt::get(DL.getIndexType(StrTy), 0)};
  Constant *FBPtr =
      ConstantExpr::getGetElementPtr(GV->getValueType(), GV, Zeros);

  // Clean up temporary PTX and fat binary files now that we've
  // successfully created the global variable.
  if (!KeepIntermediateFiles) {
    std::error_code EC;
    if ((EC = llvm::sys::fs::remove(PTXFileName)))
      errs() << "error removing PTX file: " << EC.message() << "\n";
    if ((EC = llvm::sys::fs::remove(FBFileName)))
      errs() << "error removing fatbinary file: " << EC.message() << "\n";
  }

  return FBPtr;
}

Function *CudaLoop::createCudaDtor() {

  assert(GpuBinaryHandle != nullptr &&
         "GPU binary must be created prior to codegen of dtor!");

  // No need for destructor if we don't have a handle to unregister.
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  Type *VoidPtrPtrTy = VoidPtrTy->getPointerTo();

  // void __cudaUnregisterFatBinary(void ** handle);
  FunctionCallee UnregisterFatbinFunc =
      M.getOrInsertFunction("__cudaUnregisterFatBinary",
                            FunctionType::get(VoidTy, VoidPtrPtrTy, false));

  Function *ModuleDtorFunc =
      Function::Create(FunctionType::get(VoidTy, VoidPtrTy, false),
                       GlobalValue::InternalLinkage, "__cuda_module_dtor", &M);

  BasicBlock *DtorEntryBB = BasicBlock::Create(Ctx, "entry", ModuleDtorFunc);
  IRBuilder<> DtorBuilder(DtorEntryBB);

  Value *HandleValue = DtorBuilder.CreateAlignedLoad(
      VoidPtrPtrTy, GpuBinaryHandle, DL.getPointerABIAlignment(0));
  DtorBuilder.CreateCall(UnregisterFatbinFunc, HandleValue);
  DtorBuilder.CreateRetVoid();
  return ModuleDtorFunc;
}

Function *CudaLoop::createCudaCtor(Constant *FBPtr) {
  LLVM_DEBUG(dbgs() << "cuabi : creating loop-centric Cuda Ctor.\n");

  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  Type *IntTy = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  PointerType *CharPtrTy = PointerType::getUnqual(Type::getInt8Ty(Ctx));

  Function *ModuleCtorFunc =
      Function::Create(FunctionType::get(VoidTy, VoidPtrTy, false),
                       GlobalValue::InternalLinkage, "__cuabi_module_ctor", &M);
  BasicBlock *CtorEntryBB = BasicBlock::Create(Ctx, "entry", ModuleCtorFunc);
  IRBuilder<> CtorBuilder(CtorEntryBB);

  CtorBuilder.CreateCall(KitCudaInitFn, {});

  const DataLayout &DL = M.getDataLayout();

  // Create a wrapper structure that points to the loaded GPU (fat) binary.
  StructType *FBWTy = StructType::get(IntTy,      // magic ID value
                                      IntTy,      // version (must be 1 for now)
                                      VoidPtrTy,  // "raw" gpu binary
                                      VoidPtrTy); // unused in version 1
  Constant *FBWrapperC = ConstantStruct::get(FBWTy,
                           ConstantInt::get(IntTy, FATBINARY_MAGIC_ID),
                           ConstantInt::get(IntTy, 1),
                           FBPtr,
                           ConstantPointerNull::get(VoidPtrTy));
  GlobalVariable *FBWrapper =
          new GlobalVariable(M, FBWTy, true,
                             GlobalValue::InternalLinkage,
                             FBWrapperC, "__cuabi_cufatbin_wrapper");
  FBWrapper->setSection(".nvFatBinSegment");
  FBWrapper->setAlignment(Align(DL.getPrefTypeAlignment(FBWrapper->getType())));

  FunctionCallee RegisterFatbinFunc =
      M.getOrInsertFunction("__cudaRegisterFatBinary",
                            FunctionType::get(VoidPtrPtrTy, VoidPtrTy, false));
  CallInst *RegisterFatbinCall = CtorBuilder.CreateCall(RegisterFatbinFunc,
                          CtorBuilder.CreateBitCast(FBWrapper, VoidPtrTy));

  CtorBuilder.CreateCall(KitCudaCtxCheckFn, {});

  GpuBinaryHandle = new GlobalVariable(M, VoidPtrPtrTy,
                                       false, GlobalValue::InternalLinkage,
                                       ConstantPointerNull::get(VoidPtrPtrTy),
                                       "__cuabi_gpubin_handle");
  GpuBinaryHandle->setAlignment(Align(DL.getPointerABIAlignment(0)));
  CtorBuilder.CreateAlignedStore(RegisterFatbinCall, GpuBinaryHandle,
                                 DL.getPointerABIAlignment(0));
  GpuBinaryHandle->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

  Value *GBPtr = CtorBuilder.CreateLoad(VoidPtrPtrTy,
                                        GpuBinaryHandle,
                                        "_cuabi_binh");

  llvm::Type *RegisterVarParams[] = {
    VoidPtrPtrTy, // gpu binary handle
    VoidPtrTy,    // host var
    CharPtrTy,    // device addr
    CharPtrTy,    // device name
    IntTy,        // ext
    Int64Ty,      // size
    IntTy,        // constant
    IntTy};       // global

  FunctionCallee RegisterVarFunc =
        M.getOrInsertFunction("__cudaRegisterVar",
        FunctionType::get(VoidTy, RegisterVarParams, false));


  for (auto const &GV : GVarList) {
    // TODO: the pair is invalid here (perhaps because of a 2nd level of
    // cloning when the loop gets outlined...  So, we are forced to
    // look up the device-side global by name here vs. use the pair...
    std::string SymbolName = GV->getName().str() + "_devvar";
    GlobalValue *DV = KernelModule.getNamedValue(SymbolName);
    assert(GV != nullptr && DV != nullptr &&
           "unexpected null global variable!");
    LLVM_DEBUG(dbgs() << "\t\thost: " << GV->getName()
                      << ".\n");
    LLVM_DEBUG(dbgs() << "\t\tdevice: " << DV->getName()
                      << ".\n");

    Constant *GVarName = createConstantStr(GV->getName().str());
    Constant *DVarName = createConstantStr(DV->getName().str());
    Value *GVPtr = CtorBuilder.CreatePointerCast(GV, VoidPtrTy);
    Constant *GVSize = ConstantInt::get(IntTy,
                            DL.getTypeAllocSize(GV->getValueType()));

    llvm::Value *RegVarArgs[] = {
        GBPtr,       // fat binary handle
        GVPtr,       // host-side var
        GVarName,    // device address
        DVarName,    // device var name
        ConstantInt::get(IntTy, GV->hasExternalWeakLinkage()),
        ConstantInt::get(Int64Ty, DL.getTypeAllocSize(GV->getType())),
        ConstantInt::get(IntTy, GV->isConstant()),
        ConstantInt::get(IntTy, 0)};
    CtorBuilder.CreateCall(RegisterVarFunc, RegVarArgs);
  }

  CtorBuilder.CreateCall(KitCudaCtxCheckFn, {});

  FunctionCallee RegisterFatbinEndFunc =
      M.getOrInsertFunction("__cudaRegisterFatBinaryEnd",
                            FunctionType::get(VoidTy, VoidPtrPtrTy, false));
  CtorBuilder.CreateCall(RegisterFatbinEndFunc, RegisterFatbinCall);

  CtorBuilder.CreateCall(KitCudaCtxCheckFn, {});

  if (Function *CleanupFn = createCudaDtor()) {
    // extern "C" int atexit(void (*f)(void));
    FunctionType *AtExitTy =
        FunctionType::get(IntTy, CleanupFn->getType(), false);
    FunctionCallee AtExitFunc =
        M.getOrInsertFunction("atexit", AtExitTy, AttributeList());
    CtorBuilder.CreateCall(AtExitFunc, CleanupFn);
  }

  CtorBuilder.CreateRetVoid();
  return ModuleCtorFunc;
}

void CudaLoop::bindGlobalVars(Value *CM, IRBuilder<> &B) {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);

  LLVM_DEBUG(dbgs() << "cuabi: binding " << GVarList.size()
                    << " global variables.\n");

  for(auto const& HostGV : GVarList) {
    std::string SymbolName = HostGV->getName().str() + "_devvar";

    LLVM_DEBUG(dbgs() << "\tbinding:\n"
                      << "\t\thost  : " << HostGV->getName() << "\n"
                      << "\t\tdevice: " << SymbolName << "\n");

    Constant *DevSymName = createConstantStr(SymbolName);
    DevSymName->dump();
    // NOTE: We need to use the runtime to look up the device-side
    // pointer for the named global variable.  This requires a
    // corresponding fat binary image to be created as well as an
    // associated  CUDA module (CM is this module).
    B.CreateCall(KitCudaCtxCheckFn, {});
    LLVM_DEBUG(dbgs() << "\t\tgen global symbol lookup...\n");
    Value *GVDevPtr = B.CreateCall(KitCudaGetGlobalSymbolFn,
                                  {DevSymName, CM});
    GVDevPtr->dump();
    B.CreateCall(KitCudaCtxCheckFn, {});
    // Now copy the global from the host to the device.
    Value *HostGVPtr = B.CreatePointerCast(HostGV, VoidPtrTy);
    LLVM_DEBUG(dbgs() << "\t\tgen host-> device memcpy...\n");
    B.CreateCall(KitCudaMemcpySymbolToDeviceFn,
              { HostGVPtr, GVDevPtr,
                ConstantInt::get(Int64Ty,
                      DL.getTypeAllocSize(HostGV->getValueType()))
              });
    LLVM_DEBUG(dbgs() << "\tfinished global var binding.\n");
    B.CreateCall(KitCudaCtxCheckFn, {});
  }
  LLVM_DEBUG(dbgs() << "finished binding all global vars.\n");
}

void CudaLoop::processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                                       DominatorTree &DT) {

  if (TL.getLoop()->getParentLoop() != nullptr) {
    LLVM_DEBUG(dbgs() << "process nested loop (outline).\n");
  } else {
    LLVM_DEBUG(dbgs() << "process top-level loop (outline).\n");
  }

  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);

  Function *Parent = TOI.ReplCall->getFunction();
  Value *TripCount = OrderedInputs[0];
  BasicBlock *RCBB = TOI.ReplCall->getParent();
  BasicBlock *NBB = RCBB->splitBasicBlock(TOI.ReplCall);
  TOI.ReplCall->eraseFromParent();

  IRBuilder<> B(&NBB->front());

  //runOptimizationPasses(KernelModule, OptLevel);
  transformForPTX();

  BasicBlock &EBB = Parent->getEntryBlock();
  IRBuilder<> EB(&EBB.front());

  /*

  TODO: This is broken -- don't use it or bad things happen...

  if (DefaultBlocksPerGrid != 0) {
    // TODO: At present, the runtime's kernel launch parameters can be
    // overriden in a per compiler invocation right now.  We should
    // probably ponder a way to make this work per loop (via metadata?).
    //
    // NOTE: These parameters only "stick" for the next kernel launch.
    Value *BPG = ConstantInt::get(Int32Ty, DefaultBlocksPerGrid);
    Value *TPB = ConstantInt::get(Int32Ty, DefaultThreadsPerBlock);
    EB.CreateCall(KitCudaSetDefaultLaunchParamsFn, { BPG, TPB});
  } else {
    Value *TPB = ConstantInt::get(Int32Ty, DefaultThreadsPerBlock);
    EB.CreateCall(KitCudaSetDefaultTBPFn, {TPB});
  }
  */

  ArrayType *ArrayTy = ArrayType::get(VoidPtrTy, OrderedInputs.size());
  Value *ArgArray = B.CreateAlloca(ArrayTy);
  unsigned int i = 0;
  for (Value *V : OrderedInputs) {
    Value *VP = B.CreateAlloca(V->getType());
    B.CreateStore(V, VP);
    Value *VoidVPtr = B.CreateBitCast(VP, VoidPtrTy);
    Value *ArgPtr = B.CreateConstInBoundsGEP2_32(ArrayTy, ArgArray, 0, i++);
    B.CreateStore(VoidVPtr, ArgPtr);

    // TODO: This is still experimental and currently just about as simple
    // of an approach as we can take on issuing prefetch calls.  Is it
    // better than nothing -- it is obviously very far from ideal placement.
    if (CodeGenDisablePrefetch == false) {
      Type *VT = V->getType();
      if (VT->isPointerTy()) {
        Value *VoidPP = B.CreateBitCast(V, VoidPtrTy);
        B.CreateCall(KitCudaMemPrefetchFn, {VoidPP});
      }
    }
  }
  const DataLayout &DL = M.getDataLayout();

  Value *GrainSize = TL.getGrainsize() ? ConstantInt::get(TripCount->getType(),
                                                          TL.getGrainsize())
                                       : OrderedInputs[2];
  Value *RunSizeQ = B.CreateUDiv(TripCount, GrainSize, "run_size");
  Value *RunRem = B.CreateURem(TripCount, GrainSize, "run_rem");
  Value *isRem = B.CreateICmp(ICmpInst::ICMP_UGT, RunRem,
                              ConstantInt::get(RunRem->getType(), 0));
  Value *isRemAdd = B.CreateZExt(isRem, RunSizeQ->getType());
  Value *RunSize = B.CreateZExt(B.CreateAdd(RunSizeQ, isRemAdd), Int64Ty);
  Value *argsPtr = B.CreateConstInBoundsGEP2_32(ArrayTy, ArgArray, 0, 0);

  // Generate a call to launch the kernel.
  Constant *KNameCS = ConstantDataArray::getString(Ctx, KernelName);
  GlobalVariable *KNameGV =
      new GlobalVariable(M, KNameCS->getType(), true,
                         GlobalValue::PrivateLinkage, KNameCS, ".str");
  KNameGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  Type *StrTy = KNameGV->getType();
  Constant *Zeros[] = {ConstantInt::get(DL.getIndexType(StrTy), 0),
                       ConstantInt::get(DL.getIndexType(StrTy), 0)};
  Constant *KNameParam =
      ConstantExpr::getGetElementPtr(KNameGV->getValueType(), KNameGV, Zeros);
  Constant *FBPtr = createKernelBuffer();

  Function *CudaCtorFn = createCudaCtor(FBPtr);
  // TODO: The following block could (should?) move into createCudaCtor...
  if (CudaCtorFn) {
    LLVM_DEBUG(dbgs() << "\tcuabi: adding cuda ctor() to global ctors.\n");
    Type *VoidTy = Type::getVoidTy(Ctx);
    FunctionType *CtorFTy = FunctionType::get(VoidTy, false);
    Type *CtorPFTy =
        PointerType::get(CtorFTy, M.getDataLayout().getProgramAddressSpace());
    appendToGlobalArray("llvm.global_ctors", M,
                        ConstantExpr::getBitCast(CudaCtorFn, CtorPFTy), 65536,
                        nullptr);
  }

  Value *Stream;
  if (GVarList.empty()) {
    Stream = B.CreateCall(KitCudaLaunchFn,
                          {FBPtr, KNameParam, argsPtr, RunSize},
                          "stream");
  } else {
    // If we have global variables to process we first create
    // a CUDA module assocaited with the fat binary.  Next we
    // must generate the code to bind the host and device
    // variables.  This process must copy host-side values
    // to the device prior to the kernel launch.  Once that
    // code is generated we launch the kernel via the created
    // module.
    Value *CM = B.CreateCall(KitCudaCreateFBModuleFn, {FBPtr});
    CM->dump();
    bindGlobalVars(CM, B);
    Stream = B.CreateCall(KitCudaLaunchModuleFn,
                          {CM, KNameParam, argsPtr, RunSize},
                          "stream");
  }
  B.CreateCall(KitCudaWaitFn, Stream);
}

CudaABI::CudaABI(Module &M) : TapirTarget(M) { }

CudaABI::~CudaABI() { }

void CudaABI::addToPTXFileList(const std::string &FN) {
  PTXFileList.push_back(FN);
}

Value *CudaABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  // TODO: Some quick checks suggest we are almost always getting
  // called to select this value (at least when trip counts can't
  // be easily determined?).  More work needs to go into picking
  // this value that likely strongly correlates with the launch
  // parameters...  More parameter studies are likely a key piece
  // of this puzzle.
  Value *Grainsize =
      ConstantInt::get(GrainsizeCall->getType(), DefaultGrainSize);
  // Replace uses of grainsize intrinsic call with a computed
  // grainsize value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  return Grainsize;
}

void CudaABI::lowerSync(SyncInst &SI) { /* no-op */
}

void CudaABI::addHelperAttributes(Function &F) { /* no-op */
}

void CudaABI::preProcessFunction(Function &F, TaskInfo &TI,
                                 bool OutliningTapirLoops) { /* no-op */
}

void CudaABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {
  if (!OutliningTapirLoops)
    return;
}

void CudaABI::postProcessHelper(Function &F) { /* no-op */
}

void CudaABI::preProcessOutlinedTask(llvm::Function &, llvm::Instruction *,
                                     llvm::Instruction *, bool,
                                     BasicBlock *) { /* no-op */
}

void CudaABI::postProcessOutlinedTask(Function &F, Instruction *DetachPt,
                                      Instruction *TaskFrameCreate,
                                      bool IsSpawner,
                                      BasicBlock *TFEntry) { /* no-op */
}

void CudaABI::postProcessRootSpawner(Function &F,
                                     BasicBlock *TFEntry) { /* no-op */
}

void CudaABI::processSubTaskCall(TaskOutlineInfo &TOI,
                                 DominatorTree &DT) { /* no-op */
}

void CudaABI::preProcessRootSpawner(llvm::Function &,
                                    BasicBlock *TFEntry) { /* no-op */
}

CudaABIOutputFile CudaABI::postProcessPTXFiles(Module &TM) {

  // For each parallel construct (e.g., loops) within a
  // module we create a transformed kernel as PTX source.
  // For each module we want to create a single fatbinary
  // image for the complete set of kernels.  This isn't
  // strictly required to get the code to run but is
  // required if we want compnents from the CUDA toolchain
  // to work (e.g., cuobjdump).  We also suspect some of
  // calls into the CUDA runtime behave better if this
  // binary format is used (even though it is technically
  // not documented).  The fact we run later in the pipeline
  // over other CUDA codegen paths make things slightly
  // different here...
  LLVM_DEBUG(dbgs() << "cuabi: post processing PTX files for "
                    << "module '" << TM.getName() << "'.\n");

  std::error_code EC;
  auto PTXASExe = sys::findProgramByName("ptxas");
  if ((EC = PTXASExe.getError()))
    report_fatal_error("'ptxas' not found. "
                       "Is a CUDA installation in your path?");

  // We'll create an "assembled" PTX file that is named the
  // same as the LLVM module we've just processed.
  SmallString<255> AsmFileName(Twine("__cuabi_" + TM.getName()).str());
  sys::path::replace_extension(AsmFileName, ".s");
  std::unique_ptr<ToolOutputFile> AsmFile;
  AsmFile = std::make_unique<ToolOutputFile>(AsmFileName, EC,
                                             sys::fs::OpenFlags::OF_None);

  opt::ArgStringList PTXASArgList;
  PTXASArgList.push_back(PTXASExe->c_str());
  // TODO: Not sure if we want a relocatable object or not?
  //PTXASArgList.push_back("-c");
  // --machine <bits>: Specify 32-bit vs. 64-bit host architecture.
  PTXASArgList.push_back("--machine"); // host (32- vs. 64-bit)
  PTXASArgList.push_back(HostMArch.c_str());
  // --gpu-name <gpu name>: Specify name of GPU to generate code for.
  // (e.g., 'sm_70','sm_72','sm_75','sm_80','sm_86', 'sm_87')
  PTXASArgList.push_back("--gpu-name"); // target gpu architecture.
  PTXASArgList.push_back(GPUArch.c_str());
  if (Verbose)
    // --verbose: prints code generation statistics.
    PTXASArgList.push_back("--verbose");
  if (Debug) {
    // TODO: It currently isn't possible to use both debug
    // and an optimization flag w/ ptxas -- need to check for
    // this case.
    PTXASArgList.push_back("--device-debug");
  }
  if (OptLevel > 3) {
      errs() << "warning -- cuda abi transform: "
             << "unknown optimization level.\n"
             << "\twill use level 3 instead.\n";
      OptLevel = 3;
  }
  PTXASArgList.push_back("--opt-level");
  switch (OptLevel) {
  case 0:
    PTXASArgList.push_back("0");
    break;
  case 1:
    PTXASArgList.push_back("1");
    break;
  case 2:
    PTXASArgList.push_back("2");
    break;
  case 3:
    PTXASArgList.push_back("3");
    break;
  default:
    llvm_unreachable_internal("unhandled/unexpected optimization level",
                              __FILE__, __LINE__);
    break;
  }

  if (OptLevel < 2 && AllowExpensiveOpts) {
    PTXASArgList.push_back("--allow-expensive-optimizations");
    PTXASArgList.push_back("true");
  }
  if (DisableFMA) {
    PTXASArgList.push_back("--fmad");
    PTXASArgList.push_back("false");
  }
  if (DisableConstantBank)
    PTXASArgList.push_back("--disable-optimizer-constants");

  LLVM_DEBUG(dbgs() << "\tptxas output file: "
                    << AsmFile->getFilename().str()
                    << "\n");

  PTXASArgList.push_back("--output-file");
  std::string SCodeFilename = AsmFile->getFilename().str();
  PTXASArgList.push_back(SCodeFilename.c_str());
  for (auto &FN : PTXFileList) {
    PTXASArgList.push_back(FN.c_str());
  }
  PTXASArgList.push_back(nullptr);

  // Build argv for exec'ing ptxas...
  SmallVector<const char *, 128> PTXASArgv;
  PTXASArgv.append(PTXASArgList.begin(), PTXASArgList.end());
  PTXASArgv.push_back(nullptr);
  auto PTXASArgs = toStringRefArray(PTXASArgv.data());

  LLVM_DEBUG(dbgs() << "\tptxas command line:\n";
             unsigned c = 0;
             for (auto dbg_arg : PTXASArgs) {
               dbgs() << "\t\t" << c << ": " << dbg_arg << "\n";
               c++;
             }
             dbgs() << "\n\n";
             );

  LLVM_DEBUG(dbgs() << "\texecuting ptxas...\n");
  std::string ErrMsg;
  bool ExecFailed;
  int ExecStat = sys::ExecuteAndWait(*PTXASExe, PTXASArgs, None, {},
                                     0, /* secs to wait -- 0 --> unlimited */
                                     0, /* memory limit -- 0 --> unlimited */
                                     &ErrMsg, &ExecFailed);
  if (ExecFailed)
    report_fatal_error("fatal error: 'ptxas' execution failed!");

  if (ExecStat != 0)
    // 'ptxas' ran but returned an error state.
    report_fatal_error("fatal error: 'ptxas' failure: " + ErrMsg);

  AsmFile->keep();
  return AsmFile;
}

CudaABIOutputFile CudaABI::postProcessAsmFile(CudaABIOutputFile &AsmFile,
                                              Module &TM) {

  std::error_code EC;

  SmallString<255> FatbinFilename(AsmFile->getFilename());
  sys::path::replace_extension(FatbinFilename, ".fbin");
  CudaABIOutputFile FatbinFile;
  FatbinFile = std::make_unique<ToolOutputFile>(FatbinFilename, EC,
                                            sys::fs::OpenFlags::OF_None);

  // TODO: LLVM docs suggest we shouldn't be using findProgramByName()...
  auto FatbinaryExe = sys::findProgramByName("fatbinary");
  if (EC = FatbinaryExe.getError())
    report_fatal_error("'fatbinary' not found. "
                       "Is a CUDA installation in your path?");

  opt::ArgStringList FatbinaryArgList;
  FatbinaryArgList.push_back(FatbinaryExe->c_str());

  if (HostMArch == "32")
    FatbinaryArgList.push_back("--32");
  else if (HostMArch == "64")
    FatbinaryArgList.push_back("--64");

  FatbinaryArgList.push_back("--create");
  FatbinaryArgList.push_back(FatbinFilename.c_str());

  std::string FatbinaryImgArgs = "--image=profile=" + GPUArch +
                                 ",file=" + AsmFile->getFilename().str();
  FatbinaryArgList.push_back(FatbinaryImgArgs.c_str());

std::list<std::string> PTXFilesArgList;
if (EmbedPTXInFatbinaries) {
    std::string VArchStr = virtualArchForCudaArch(GPUArch);
    if (VArchStr == "unknown")
      report_fatal_error("cuabi: no virtual target for given gpuarch '"
                         + GPUArch + "'!");

    std::string PTXFixedArgStr = "--image=profile=" + VArchStr + ",file=";
    for (auto &PTXFile : PTXFileList) {
      std::string arg = PTXFixedArgStr + PTXFile;
      std::list<std::string>::const_iterator it;
      it = PTXFilesArgList.emplace(PTXFilesArgList.end(), std::move(arg));
      FatbinaryArgList.push_back(it->c_str());
    }
  }
  FatbinaryArgList.push_back(nullptr);

  SmallVector<const char *, 128> FatbinaryArgv;
  FatbinaryArgv.append(FatbinaryArgList.begin(), FatbinaryArgList.end());
  auto FatbinaryArgs = toStringRefArray(FatbinaryArgv.data());

  LLVM_DEBUG(dbgs() << "\tfatbinary command line:\n";
             unsigned c = 0; for (auto dbg_arg
                                  : FatbinaryArgs) {
               dbgs() << "\t\t" << c << ": " << dbg_arg << "\n";
               c++;
             } dbgs() << "\n\n";);

  std::string ErrMsg;
  bool ExecFailed;
  int ExecStat = sys::ExecuteAndWait(*FatbinaryExe, FatbinaryArgs, None, {},
                                     0, /* secs to wait -- 0 --> unlimited */
                                     0, /* memory limit -- 0 --> unlimited */
                                     &ErrMsg, &ExecFailed);
  if (ExecFailed)
    report_fatal_error("unable to execute 'fatbinary'.");

  if (ExecStat != 0)
    // 'fatbinary' ran but returned an error state.
    // TODO: Need to check what sort of actual state 'fatbinary'
    // returns to the environment -- currently assuming it
    // matches standard practices...
    report_fatal_error("'fatbinary' error:" + ErrMsg);

  if (EmbedPTXInFatbinaries) {
    std::list<std::string>::iterator it = PTXFilesArgList.begin();
    while(it != PTXFilesArgList.end()) {
      PTXFilesArgList.erase(it++);
    }
  }

  FatbinFile->keep();
  return FatbinFile;
}

void CudaABI::postProcessModule(Module &TM) {

  LLVM_DEBUG(dbgs() << "cuabi: post processing module '"
                    << TM.getName() << "'\n");
  LLVM_DEBUG(dbgs() << "\tinternally referenced module is '"
                    << M.getName() << "'\n");

  if (!PTXFileList.empty()) {
    CudaABIOutputFile AsmFile = postProcessPTXFiles(TM);
    CudaABIOutputFile FatbinFile = postProcessAsmFile(AsmFile, TM);
    //postProcessEmbedFatbinary(FatbinFile, TM);
  }

}

LoopOutlineProcessor *
CudaABI::getLoopOutlineProcessor(const TapirLoopInfo *TL) {
  // Module names can be a bit confusing in the transformation.
  // We base the name of the module that we will hold the
  // CUDA kernel(s) in after the parent module that has the
  // loop that has been outlined.  In this case 'M' is this
  // parent module and KernelModule is the new Module we will
  // create to hold the kernel and other associated code...
  std::string ModuleName = M.getName().str();
  // PTX dislikes names containing '.' -- replace them with
  // underscores.
  std::replace(ModuleName.begin(), ModuleName.end(), '.', '_');
  std::replace(ModuleName.begin(), ModuleName.end(), '-', '_');
  std::string KN;
  bool MakeKNUnique = true;
  if (M.getNamedMetadata("llvm.dbg.cu") || M.getNamedMetadata("llvm.dbg")) {
    unsigned LineNumber = TL->getLoop()->getStartLoc()->getLine();
    KN = CUABI_KERNEL_NAME_PREFIX + ModuleName + "_" + Twine(LineNumber).str();
    MakeKNUnique = false;
  } else
    KN = CUABI_KERNEL_NAME_PREFIX + ModuleName;
  CudaLoop *CLOP = new CudaLoop(M, KN, this, MakeKNUnique);
  return CLOP;
}
