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
#include "kitsune/Core/BasicBlockUtils.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/ModuleAttrs.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "llvm/IR/Intrinsics.h"

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

    if (tto.getHipXnack() == MaybeBool::On) {
      LLVM_DEBUG(dbgs() << "\t\tenable xnack via ctor runtime call.\n");
      builder.CreateIntrinsic(Intrinsic::kit_runtime_set_xnack, {ctt});
    }

    if (genCtorOpts.useYLaunch) {
      LLVM_DEBUG(
          dbgs()
          << "\t\tenable y-axis launch pattern via ctor runtime call.\n");
      builder.CreateIntrinsic(Intrinsic::kit_runtime_set_y_axis_kernel_launch,
                              {ctt});
    }
  }

public:
  GenerateCtorHip(const TTOptions &tto,
                  const detail::GenerateCtorOptions &genCtorOpts)
      : GenerateCtorGPU(TTID::Hip, tto, genCtorOpts) {}
};

class GenerateCtorsImpl {
private:
  const TTOptions &tto;
  const detail::GenerateCtorOptions &genCtorOpts;

private:
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
      // should not even get here, but it's ok if we do.
      return;
    default:
      llvm_unreachable("generateCtor: TTID not handled");
    }
  }

public:
  GenerateCtorsImpl(const TTOptions &tto,
                    const detail::GenerateCtorOptions &genCtorOpts)
      : tto(tto), genCtorOpts(genCtorOpts) {}

  void run(Module &m) {
    if (std::optional<SmallSet<TTID, 0>> tts = getTTsAttr(m))
      for (TTID tt : *tts)
        generateCtorDtor(m, tt);
    else
      emitDiagnostic(DiagID::WarnGeneric,
                     "kit.module.tts attribute not found in module. No ctors "
                     "will be generated");
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
  detail::GenerateCtorOptions genCtorOpts;
  if (&clUseYLaunch)
    genCtorOpts.useYLaunch = clUseYLaunch;

  GenerateCtorsImpl(ttOpts, genCtorOpts).run(m);

  // This never invalidates any analyses since, at most, only the initializer of
  // a global variable will have changed. The generated ctors will not be called
  // explicitly in the code, so the callgraph will not have changed either.
  return PreservedAnalyses::all();
}
