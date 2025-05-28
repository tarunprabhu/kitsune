//=- GenerateKitsuneCtor.cpp - Generate global ctors for Kitsune --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of pass that generates global constructors for Kitsune's
// runtime
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Kitsune/GenerateKitsuneCtors.h"
#include "kitsune/Config/config.h"
#include "llvm/Analysis/TapirTargetAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Frontend/Tapir/Tapir.h"
#include "llvm/Frontend/Tapir/TapirTargetOptions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/KitsuneMetadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/Transforms/Utils/KitsuneUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#define DEBUG_TYPE "generate-kitsune-ctors"

using namespace llvm;

// Request that the runtime carry out an extra set of steps to attempt to
// refine the launch parameters of kernels.  In this mode of operation the
// compiler will provide some compile-time information onto the runtime for
// assisting in the analysis an refinement of launches.
static cl::opt<bool> clRefineLaunches(
    "cuabi-refine-launches", cl::init(true), cl::Hidden,
    cl::desc("Enable runtime's refinement of launch parameters"));

namespace {

using GetTLI = std::function<TargetLibraryInfo &(Function &)>;

/// Helper class to generate a ctor for kitcuda (Kitsune's runtime for the cuda
/// tapir target). This will also create variables for the fat binary and the
/// bundle containing the fat binary that is needed by the cuda runtime.
///
/// Registering the fat binary image (and all the associated components) is
/// an undocumented portion of the CUDA API. One place to peek for some details
/// hides in the cuda header files; specifially fatbinary_section.h. This shows
/// the following struct that we need to have in the host side code.
///
///    struct fatbinC_Wrapper_t {
///      int magic;
///      int version;
///      const unsigned long long *data;
///      void *filename_or_fatbins;
///    };
///
/// * Per the header, the magic number is 0x466243B1
/// * FATBINC_VERSION is 1 and FATBINC_LINK_VERSION is 2 (more below)
/// * Then section and segments are needed that contains the "fatbin control
///   structure".  This loosely looks like:
///
///        Control section name: ".nvFatBinSegment"
///        Fatbinary section name: ".nv_fatbin"
///        Pre-linked relocatable section: "__nv_relfatbin"
///
/// * The last struct member varies between versions. In the case of version 1
///   it can be a offline filename and for version 2 it is an array of
///   pre-linked fatbins.
///
class GenerateKitsuneCudaCtor {
private:
  static constexpr int magic = 0x466243B1;
  static constexpr int version = 1;
  static constexpr const char *controlSectionName = ".nvFatBinSegment";

private:
  GetTLI getTLI;
  const TapirTargetOptions &tto;

private:
  /// Create a global variable containing the fat binary "bundle". This
  /// consists of the fat binary and some metadata.
  GlobalVariable *createBundleGV(Module &m, GlobalVariable *fatBin) {
    const DataLayout &dl = m.getDataLayout();

    LLVMContext &ctx = m.getContext();
    Type *i32Ty = Type::getInt32Ty(ctx);
    PointerType *ptrTy = PointerType::getUnqual(ctx);
    Type *idxTy = dl.getIndexType(ptrTy);
    StructType *bundleTy = StructType::get(/*magic*/ i32Ty,
                                           /*version*/ i32Ty,
                                           /*fat binary data*/ ptrTy,
                                           /*unused*/ ptrTy);

    Constant *zero = ConstantInt::get(idxTy, 0);
    Constant *zeros[] = {zero, zero};

    // Wrap the fatbinary in struct that the CUDA runtime and tools expect
    // to exist in final objects/executables.
    Constant *bundleInit = ConstantStruct::get(
        bundleTy, ConstantInt::get(i32Ty, magic),
        ConstantInt::get(i32Ty, version),
        ConstantExpr::getGetElementPtr(fatBin->getValueType(), fatBin, zeros),
        ConstantPointerNull::get(ptrTy));

    GlobalVariable *g = new GlobalVariable(m, bundleTy, /*isConstant*/ true,
                                           GlobalValue::InternalLinkage,
                                           bundleInit, ".kitcuda.bundle");
    g->setSection(controlSectionName);
    g->setAlignment(dl.getPrefTypeAlign(g->getType()));

    return g;
  }

  /// Create a global variable that will contain the "handle" to the fat binary.
  /// The handle is the value returned by __cudaRegisterFatBinary(). The handle
  /// is saved into this global and read from there by the global dtor and
  /// passed to __cudaUnregisterFatBinary().
  GlobalVariable *createBundleHandleGV(Module &m) {
    const DataLayout &dl = m.getDataLayout();

    LLVMContext &ctx = m.getContext();
    PointerType *ptrTy = PointerType::getUnqual(ctx);

    GlobalVariable *g = new GlobalVariable(
        m, ptrTy, /*isConstant*/ false, GlobalValue::InternalLinkage,
        ConstantPointerNull::get(ptrTy), ".kitcuda.handle");
    g->setAlignment(dl.getPointerABIAlignment(0));
    g->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

    return g;
  }

  Function *createCtor(Module &m, const Module &devM, Function *dtor,
                       GlobalVariable *gBundle, GlobalVariable *gBundleHandle) {
    const DataLayout &dl = m.getDataLayout();
    Align ptrAlign = dl.getPointerABIAlignment(0);

    LLVMContext &ctx = m.getContext();

    Type *voidTy = Type::getVoidTy(ctx);
    PointerType *ptrTy = PointerType::getUnqual(ctx);
    Type *i32Ty = Type::getInt32Ty(ctx);
    Type *boolTy = Type::getInt8Ty(ctx);
    ConstantInt *constTT = getConstantInt(ctx, TapirTargetID::Cuda);

    FunctionType *ctorTy = FunctionType::get(voidTy, ptrTy, false);
    Function *ctor = Function::Create(ctorTy, GlobalValue::InternalLinkage,
                                      ".kitcuda.ctor", &m);

    TargetLibraryInfo &tli = getTLI(*ctor);
    Type *sizeTTy = tli.getSizeTType(m);

    IRBuilder<> builder(BasicBlock::Create(ctx, "entry", ctor));

    FunctionCallee kitrtInitialize =
        Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kitrt_initialize);
    builder.CreateCall(kitrtInitialize, {constTT});

    // Enable verbose mode early in the constructor so all verbose statements
    // are printed after the runtime has been initialized.
    FunctionCallee kitrtEnableVerbose =
        Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kitrt_enable_verbose);
    builder.CreateCall(
        kitrtEnableVerbose,
        {ConstantInt::get(boolTy, tto.getKitrtVerbose(), false)});

    FunctionCallee kitrtSetFixedTPB =
        Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kitrt_set_fixed_tpb);
    unsigned fixedTPB = tto.getFixedThreadsPerBlock();
    if (fixedTPB) {
      builder.CreateCall(kitrtSetFixedTPB,
                         {constTT, ConstantInt::get(i32Ty, fixedTPB)});
    }

    // If the MaxThreadsPerBlock has not been set, use a value of 1024 anyway.
    // At the time of writing, exceeding this value degrades performance. This
    // might change, and we may even have to set a different value depending
    // on the specific GPU architecture.
    //
    // FIXME: Don't hardcode this value here. Maybe move it to a named constant.
    FunctionCallee kitrtSetMaxTPB =
        Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kitrt_set_max_tpb);
    unsigned maxTPB = tto.getMaxThreadsPerBlock();
    if (!maxTPB)
      maxTPB = 1024;
    builder.CreateCall(kitrtSetMaxTPB,
                       {constTT, ConstantInt::get(i32Ty, maxTPB)});

    FunctionCallee kitrtEnableRefineLaunches =
        Intrinsic::getOrInsertDeclaration(
            &m, Intrinsic::kitrt_enable_refine_launches);
    builder.CreateCall(kitrtEnableRefineLaunches,
                       {constTT, ConstantInt::get(boolTy, clRefineLaunches)});

    // TODO: The parameters to the CUDA registration calls can be opaque about
    // specifics (e.g., types).  Once we sort out some details we should clean
    // this up.

    // The general layout of the calls for fat binary registration
    // look something like this:
    //
    // void** __cudaRegisterFatBinary(void *fatCubin);
    //
    // void __cudaRegisterVar(void **fatCubinHandle,
    //                        char  *hostVar,
    //                        char  *deviceAddress,
    //                        const char  *deviceName,
    //                        int    ext,
    //                        size_t size,
    //                        int    constant,
    //                        int    global);
    //
    // void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
    //
    FunctionCallee cudaRegisterFatBinary =
        getOrInsertLibFunc(&m, tli, LibFunc_cuda_register_fat_binary);
    Value *bundleHandle = builder.CreateCall(cudaRegisterFatBinary, gBundle);
    builder.CreateAlignedStore(bundleHandle, gBundleHandle, ptrAlign);

    // Register any non-constant global variables used in the kernel module.
    // TODO: It is not 100% clear what calls we actually need to make here for
    // kernel, variable, etc. registration with CUDA.  Clang makes these calls
    // but we are targeting CUDA driver API entry points via the Kitsune runtime
    // library so these calls are potentially unneeded.
    FunctionCallee cudaRegisterVar =
        getOrInsertLibFunc(&m, tli, LibFunc_cuda_register_var);

    std::vector<GlobalVariable *> gvs;
    for (const GlobalVariable& g : devM.globals())
      if (not g.isConstant())
        gvs.push_back(m.getGlobalVariable(g.getName()));

    for (GlobalVariable *gv : gvs) {
      uint64_t size = dl.getTypeAllocSize(gv->getType());
      GlobalVariable *gName = getOrCreateGlobalString(m, gv->getName());
      Value *args[] = {bundleHandle,
                       gv,
                       gName,
                       gName,
                       ConstantInt::get(i32Ty, 0), // gv->isExternalLinkage()
                       ConstantInt::get(sizeTTy, size),
                       ConstantInt::get(i32Ty, gv->isConstant()),
                       ConstantInt::get(i32Ty, 0)};

      LLVM_DEBUG(dbgs() << "\t\t\thost global '" << gv->getName()
                        << "' to device '.\n");
      builder.CreateCall(cudaRegisterVar, args);
    }

    // Wrap up fatbinary registration steps.
    FunctionCallee cudaRegisterFatBinaryEnd =
        getOrInsertLibFunc(&m, tli, LibFunc_cuda_register_fat_binary_end);
    builder.CreateCall(cudaRegisterFatBinaryEnd, bundleHandle);

    // Now add the dtor to help us clean up at program exit.
    FunctionCallee atExit = getOrInsertLibFunc(&m, tli, LibFunc_atexit);
    builder.CreateCall(atExit, dtor);

    builder.CreateRetVoid();
    return ctor;
  }

  Function *createDtor(Module &m, GlobalVariable *gBundleHandle) {
    const DataLayout &dl = m.getDataLayout();
    Align ptrAlign = dl.getPointerABIAlignment(0);

    LLVMContext &ctx = m.getContext();
    Type *voidTy = Type::getVoidTy(ctx);
    PointerType *ptrTy = PointerType::getUnqual(ctx);
    ConstantInt *constTT = getConstantInt(ctx, TapirTargetID::Cuda);

    FunctionType *dtorTy = FunctionType::get(voidTy, ptrTy, false);
    Function *dtor = Function::Create(dtorTy, GlobalValue::InternalLinkage,
                                      ".kitcuda.dtor", &m);

    TargetLibraryInfo &tli = getTLI(*dtor);
    IRBuilder<> builder(BasicBlock::Create(ctx, "entry", dtor));
    Value *handle = builder.CreateAlignedLoad(ptrTy, gBundleHandle, ptrAlign);

    FunctionCallee cudaUnregisterFatBinary =
        getOrInsertLibFunc(&m, tli, LibFunc_cuda_unregister_fat_binary);
    builder.CreateCall(cudaUnregisterFatBinary, handle);

    FunctionCallee kitrtFinalize =
        Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kitrt_finalize);
    builder.CreateCall(kitrtFinalize, {constTT});

    builder.CreateRetVoid();
    return dtor;
  }

public:
  GenerateKitsuneCudaCtor(GetTLI getTLI, const TapirTargetOptions &tto)
      : getTLI(getTLI), tto(tto) {}

  void run(Module &m) {
    GlobalVariable *gFB = getEmbeddedFB(TapirTargetID::Cuda, m);
    assert(gFB && "Could not find global with embedded cuda fat binary");

    GlobalVariable *gBC = getEmbeddedBC(TapirTargetID::Cuda, m);
    assert(gBC && "Could not find global with embedded bitcode");

    std::unique_ptr<Module> devM = parseEmbeddedBC(*gBC);

    GlobalVariable *gBundle = createBundleGV(m, gFB);
    GlobalVariable *gBundleHandle = createBundleHandleGV(m);
    Function *dtor = createDtor(m, gBundleHandle);
    Function *ctor = createCtor(m, *devM, dtor, gBundle, gBundleHandle);

    // Set the priority of this ctor to be very low so it is one of the last to
    // run.
    appendToGlobalCtors(m, ctor, 65536);
  }
};

/// Helper class to generate a ctor for kithip (Kitsune's runtime for the hip
/// tapir target).
class GenerateKitsuneHipCtor {
private:
  /// Function to get the target library info object for a function. Since the
  /// target library info can only be retrieved using a function analysis
  /// manager, we need a function in order to get this. The first time we have
  /// an LLVM function is when we create one.
  GetTLI getTLI;
  const TapirTargetOptions &tto;

private:
  /// Create a global variable containing the fat binary "bundle". This
  /// consists of the fat binary and some metadata.
  GlobalVariable *createBundleGV(Module &m, GlobalVariable *fatBin);

public:
  GenerateKitsuneHipCtor(GetTLI getTLI, const TapirTargetOptions &tto)
      : getTLI(getTLI), tto(tto) {}

  void run(Module &m) {
    // TODO: Implement this.
  }
};

} // namespace

/// Should a ctor be generated for a GPU-centric tapir target. To determine if
/// this is the case, check that at least one call to Kitsune's launch kernel
/// intrinsic is present in the module.
static bool shouldGenerateGPUCtor(Module &m, TapirTargetID tt) {
  assert((tt == TapirTargetID::Cuda || tt == TapirTargetID::Hip) &&
         "shouldGenerateGPUCtor: Tapir target must be GPU-centric");
  StringRef launch = Intrinsic::getBaseName(Intrinsic::kitrt_launch_kernel);
  if (Function *f = m.getFunction(launch)) {
    for (Use &u : f->uses()) {
      if (auto *call = dyn_cast<CallBase>(u.getUser())) {
        // Although unlikely, the intrinsic could have been passed as an
        // argument to some other function. Just in case, check that the callee
        // at this site is the launch kernel function.
        if (call->getIntrinsicID() == Intrinsic::kitrt_launch_kernel) {
          auto *cint = dyn_cast<ConstantInt>(call->getArgOperand(0));
          if (cint->getZExtValue() == unsigned(tt))
            return true;
        }
      }
    }
  }
  return false;
}

/// Should a ctor be generated for Kitsune's runtime for the cuda tapir target.
static bool shouldGenerateCudaCtor(Module &m) {
  return shouldGenerateGPUCtor(m, TapirTargetID::Cuda);
}

/// Should a ctor be generated for Kitsune's runtime for the hip tapir target.
static bool shouldGenerateHipCtor(Module &m) {
  return shouldGenerateGPUCtor(m, TapirTargetID::Hip);
}

namespace llvm {

PreservedAnalyses GenerateKitsuneCtorsPass::run(Module &m,
                                                ModuleAnalysisManager &mam) {
  // If no primary tapir target has been set, there will be nothing to do, so
  // bail out immediately.
  const TapirTargetInfo &tgi = mam.getResult<TapirTargetAnalysis>(m);
  if (not tgi.hasID())
    return PreservedAnalyses::all();

  auto &fam = mam.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();
  auto getTLI = [&](Function &f) -> TargetLibraryInfo & {
    return fam.getResult<TargetLibraryAnalysis>(f);
  };
  const TapirTargetOptions &tto = tgi.getOptions();

  if (shouldGenerateCudaCtor(m))
    GenerateKitsuneCudaCtor(getTLI, tto).run(m);

  if (shouldGenerateHipCtor(m))
    GenerateKitsuneHipCtor(getTLI, tto).run(m);

  // This never invalidates any analyses since only a global variable will have
  // changed. The generated ctors will not be called explicitly in the code,
  // so the callgraph will not have changed either.
  return PreservedAnalyses::all();
}

} // namespace llvm
