//===- GenerateCtors.cpp - Generate global ctors and dtors ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generates global constructors and destructors needed by Kitsune.
//
// These will initialize and finalize kitsune's runtime(s). They will do the
// same for any other runtimes (such as cuda and hip). These may involve
// registering global variables and fat binaries with the underlying
// GPU-specific runtime, setting environment variables etc. Not all tapir
// targets require Kitsune's runtime, but this pass will always be run when
// tapir is enabled.
//
// In addition to creating the constructors and destructors, this pass will
// also create any any global variables needed by the global ctor. In the case
// of the GPU tapir targets and associated runtimes, these include globals for
// the fat binary, the bundle that wraps the fat binary etc.
//
// This pass should only be run once per module and should be run as late as
// possible to ensure that all tapir targets have been run already.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/GenerateCtors.h"
#include "GenerateCtorsCPU.h"
#include "GenerateCtorsGPU.h"
#include "kitsune/Analysis/TTObjectsAnalysis.h"
#include "kitsune/Config/Config.h"
#include "kitsune/Core/BasicBlockUtils.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

// FIXME: We really should not be exposing command line options from other
// source files. These are experimental options that have been hacked in for the
// moment. If this is useful, we should consider adding it to the tapir target
// options instead. Otherwise, it should be removed altogether.
//
// These are declared [[weak]] because they are defined in one of the tapir
// targets. The tapir targets are not guaranteed to be built, therefore, these
// globals may not be available at link time.
//
#ifdef __APPLE__
#define WEAK weak_import
#else
#define WEAK weak
#endif
extern __attribute__((WEAK)) cl::opt<bool> clUseYLaunch;

namespace {

/// Helper class to generate a ctor for kitcuda (Kitsune's runtime for the cuda
/// tapir target). This will also create variables for the fat binary and the
/// bundle containing the fat binary that is needed by the cuda runtime.
///
/// Registering the fat binary image (and all the associated components) is
/// an undocumented portion of the CUDA API. One place to peek for some details
/// hides in the cuda header files; specifically fatbinary_section.h. This shows
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
class GenerateCtorCuda : public detail::GenerateCtorGPU {
private:
  static constexpr int magic = 0x466243B1;
  static constexpr int version = 1;
  static constexpr const char *section = ".nvFatBinSegment";

protected:
  virtual int getBundleMagic() const override { return magic; }

  virtual int getBundleVersion() const override { return version; }

  virtual StringRef getBundleSection() const override { return section; }

  virtual void genCtorAfterDevCodeRegistration(IRBuilder<> &builder,
                                               GlobalVariable *gBundleHandle,
                                               const Module &devM) override {
    Module *m = getModule(*builder.GetInsertBlock());
    LLVMContext &ctx = m->getContext();
    Align align = m->getDataLayout().getPointerABIAlignment(0);
    PointerType *ptr = PointerType::getUnqual(ctx);

    Constant *ctt = toConstant(tt, ctx);

    // Wrap up device code registration steps.
    Value *handle = builder.CreateAlignedLoad(ptr, gBundleHandle, align);
    builder.CreateIntrinsic(Intrinsic::kit_gpu_register_devcode_end,
                            {ctt, handle});
  }

public:
  GenerateCtorCuda(const TTOptions &tto,
                   const detail::GenerateCtorOptions &genCtorOpts)
      : GenerateCtorGPU(TTID::Cuda, tto, genCtorOpts) {}
};

/// Helper class to generate a ctor for kithip (Kitsune's runtime for the hip
/// tapir target).
class GenerateCtorHip : public detail::GenerateCtorGPU {
private:
  static constexpr int magic = 0x48495046;
  static constexpr int version = 1;
  static constexpr const char *section = ".hipFatBinSegment";

private:
  virtual int getBundleMagic() const override { return magic; }

  virtual int getBundleVersion() const override { return version; }

  virtual StringRef getBundleSection() const override { return section; }

  virtual void genCtorBeforeDevCodeRegistration(IRBuilder<> &builder) override {
    Module *m = getModule(*builder.GetInsertBlock());
    LLVMContext &ctx = m->getContext();

    Constant *ctt = toConstant(tt, ctx);

    if (tto.getHipXnack() == MaybeBool::On)
      LLVM_DEBUG(dbgs() << "\t\tenable xnack via ctor runtime call.\n");
    Constant *cXnack =
        toConstant(uint8_t(tto.getHipXnack() == MaybeBool::On), ctx);
    builder.CreateIntrinsic(Intrinsic::kit_runtime_set_xnack, {ctt, cXnack});

    if (genCtorOpts.useYLaunch)
      LLVM_DEBUG(
          dbgs()
          << "\t\tenable y-axis launch pattern via ctor runtime call.\n");
    Constant *cYAxisLaunch = toConstant(uint8_t(genCtorOpts.useYLaunch), ctx);
    builder.CreateIntrinsic(Intrinsic::kit_runtime_set_y_axis_kernel_launch,
                            {ctt, cYAxisLaunch});
  }

public:
  GenerateCtorHip(const TTOptions &tto,
                  const detail::GenerateCtorOptions &genCtorOpts)
      : GenerateCtorGPU(TTID::Hip, tto, genCtorOpts) {}
};

/// Implementation class to generate ctors
class GenerateCtorsImpl {
private:
  FunctionAnalysisManager &fam;
  const TTOptions &tto;
  const detail::GenerateCtorOptions &genCtorOpts;

private:
  /// Is the given intrinsic \param id called at least once in the module \param
  /// m with the tapir target id \param tt
  bool isCalledWithTTID(Module &m, Intrinsic::ID id, TTID tt) {
    if (Function *f = m.getFunction(Intrinsic::getBaseName(id)))
      for (Use &u : f->uses())
        if (auto *call = dyn_cast<CallBase>(u.getUser()))
          // Although unlikely, the intrinsic could have been passed as an
          // argument to some other function. Just in case, check that the
          // callee at this site is the launch kernel function.
          if (call->getCalledFunction() == f)
            if (auto *cint = dyn_cast<ConstantInt>(call->getArgOperand(0)))
              if (std::optional<TTID> ttid = fromConstant<TTID>(*cint))
                if (*ttid == tt)
                  return true;
    return false;
  }

  /// Check if any functions from Cilk's runtime are used.
  bool usesCilkRT(Module &m) {
    // __cilkrts_status is probably safe to check for since it is likely to be
    // used in most cases. Checking for a cilk stack frame is better, but the
    // stack frame handling functions get inlined making it difficult to be
    // certain that it is used. We could check that a stack frame is alloca'ed,
    // but that would involve checking the type name. This is not reliable,
    // especially on large codes at high optimization levels because the struct
    // type may be mangled, or replaced with an anonymous type.
    if (m.getGlobalVariable("__cilkrts_status"))
      return true;

    // There is a chance that the function will not have been inlined, so we may
    // as well check for that in case __cilkrts_status is not present.
    if (m.getFunction("__cilkrts_enter_frame"))
      return true;

    // The two cases above should handle the common case. But in case they fail,
    // attempt to find the creation of a cilk stack frame. The stack frame will
    // have been created in the entry block of the function, so just look there.
    //
    // Ideally, we should only look in the outlined function, but there is no
    // way to reliably identify such functions.
    for (Function &f : m.functions())
      if (f.size())
        for (Instruction &inst : f.getEntryBlock())
          if (auto *alloca = dyn_cast<AllocaInst>(&inst))
            if (auto *sty = dyn_cast<StructType>(alloca->getAllocatedType()))
              if (sty->hasName())
                return sty->getName().starts_with("__cilkrts_stack_frame");

    return false;
  }

  // Check if the module contains at least one serialized loop.
  bool hasSerializedLoop(Module &m) {
    for (Function &f : m.functions()) {
      if (f.size()) {
        LoopInfo &li = fam.getResult<LoopAnalysis>(f);
        for (Loop *loop : li.getLoopsInPreorder())
          if (hasSerializedAttr(*loop))
            return true;
      }
    }
    return false;
  }

  /// Should a ctor be generated for a tapir target.
  bool shouldGenerateCtor(Module &m, TTID tt) {
    switch (tt) {
    case TTID::Cuda:
    case TTID::Hip:
      return isCalledWithTTID(m, Intrinsic::kit_async_gpu_kernel_launch, tt);
    case TTID::OpenCilk:
      return usesCilkRT(m);
    case TTID::OpenMP:
    case TTID::Qthreads:
      return isCalledWithTTID(m, Intrinsic::kit_cpu_threads_launch, tt);
    case TTID::Pthreads:
      return isCalledWithTTID(m, Intrinsic::kit_async_cpu_threads_launch, tt);
    case TTID::Serial:
      return hasSerializedLoop(m);
    case TTID::Custom:
      return false;
    default:
      llvm_unreachable("shouldGenereateCtor: TTID not handled");
    }
  }

  /// Generate a ctor and dtor for the given tapir target.
  void generateCtorDtor(Module &m, TTID tt) {
    switch (tt) {
    case TTID::Cuda:
      return GenerateCtorCuda(tto, genCtorOpts).run(m);
    case TTID::Hip:
      return GenerateCtorHip(tto, genCtorOpts).run(m);
    case TTID::OpenCilk:
    case TTID::OpenMP:
    case TTID::Pthreads:
    case TTID::Qthreads:
    case TTID::Serial:
      return detail::GenerateCtorCPU(tt, tto).run(m);
    case TTID::Custom:
      // We don't generate ctor or dtor for these tapir targets. Technically, we
      // should even get here, but it's ok if we do.
      return;
    default:
      llvm_unreachable("generateCtor: TTID not handled");
    }
  }

public:
  GenerateCtorsImpl(FunctionAnalysisManager &fam, const TTOptions &tto,
                    const detail::GenerateCtorOptions &genCtorOpts)
      : fam(fam), tto(tto), genCtorOpts(genCtorOpts) {}

  void run(Module &m) {
    for (TTID tt : kitKnownTTs())
      if (shouldGenerateCtor(m, tt))
        generateCtorDtor(m, tt);
  }
};

} // namespace

PreservedAnalyses GenerateCtorsPass::run(Module &m,
                                         ModuleAnalysisManager &mam) {
  // If no primary tapir target has been set, there will be nothing to do, so
  // bail out immediately.
  const TTObjects &ttObjs = mam.getResult<TTObjectsAnalysis>(m);
  if (not ttObjs.hasTTID())
    return PreservedAnalyses::all();

  const TTOptions &ttOpts = ttObjs.getOptions();
  FunctionAnalysisManager &fam =
      mam.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();

  detail::GenerateCtorOptions genCtorOpts;
  if (&clUseYLaunch)
    genCtorOpts.useYLaunch = clUseYLaunch;

  GenerateCtorsImpl(fam, ttOpts, genCtorOpts).run(m);

  // This never invalidates any analyses since, at most, only the initializer of
  // a global variable will have changed. The generated ctors will not be called
  // explicitly in the code, so the callgraph will not have changed either.
  return PreservedAnalyses::all();
}
