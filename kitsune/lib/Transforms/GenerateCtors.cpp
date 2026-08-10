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
#include "kitsune/Analysis/TTObjectsAnalysis.h"
#include "kitsune/Core/AddrSpace.h"
#include "kitsune/Core/BasicBlockUtils.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/FuncUtils.h"
#include "kitsune/Core/IRBuilderUtils.h"
#include "kitsune/Core/LibFuncs.h"
#include "kitsune/Core/ModuleAttrs.h"
#include "kitsune/Core/TTOptions.h"
#include "kitsune/Object/Sections.h"
#include "kitsune/Shared/RTInitOptions.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "kitsune/Support/ErrorHandling.h"
#include "kitsune/Support/OstreamUtils.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#define DEBUG_TYPE "kit-ctors"

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

// The priority must be in the range [101,65535] with larger values having
// lower priority relative to other global constructors in @llvm.global_ctors
// (respectively destructors in @llvm.global_dtors).
static constexpr unsigned kitCtorPriority = 65535;
static constexpr unsigned kitDtorPriority = 65535;

namespace {

/// Options to generate global ctors for kitsune's runtime.
struct GenerateCtorOptions {
public:
  /// Launch kernel using Y-axis threading.
  unsigned useYLaunch : 1;
};

/// Base class to generate ctors/dtors for GPU-centric tapir targets. The
/// default implementation should cover most cases for the cuda and hip tapir
/// targets. Aside from the pure virtual methods that must be implemented by
/// subclasses, the two other useful methods that may be overridden are
/// genCtorBeforeDevCodeRegistration() and genCtorAfterDevCodeRegistration().
/// Most calls that setup the runtime should be added to the first of these.
/// genCtorAfterDevCodeRegister is only really necessary if one has to call the
/// runtime to indicate that the device code has been registered, though there
/// may be work that needs to be done after the device code has been registered.
///
/// NOTE: The default implementations of these methods do nothing.
class PopulateCtorGPU {
protected:
  TTID tt;
  const TTOptions &tto;
  const GenerateCtorOptions &genCtorOpts;

protected:
  PopulateCtorGPU(TTID tt, const TTOptions &tto,
                  const GenerateCtorOptions &genCtorOpts)
      : tt(tt), tto(tto), genCtorOpts(genCtorOpts) {}

  /// Create a global variable containing the fat binary "bundle". This
  /// consists of the device code and some metadata.
  virtual GlobalVariable *createBundleGV(Module &m, GlobalVariable *devCode) {
    const DataLayout &dl = m.getDataLayout();
    LLVMContext &ctx = m.getContext();

    Type *i32 = Type::getInt32Ty(ctx);
    PointerType *ptr = PointerType::getUnqual(ctx);
    Type *idxTy = dl.getIndexType(ptr);
    StructType *bundleTy = StructType::get(i32 /*magic*/, i32 /*version*/,
                                           ptr /*device code*/, ptr /*unused*/);

    // TODO: Do we really need the ConstantExpr here or can we just pass the
    // global variable directly?
    Constant *zero = ConstantInt::get(idxTy, 0);
    Constant *zeros[] = {zero, zero};
    Constant *offset =
        ConstantExpr::getGetElementPtr(devCode->getValueType(), devCode, zeros);

    Constant *magic = ConstantInt::get(i32, getBundleMagic());
    Constant *version = ConstantInt::get(i32, getBundleVersion());
    Constant *cnull = ConstantPointerNull::get(ptr);

    // Wrap the device code in a struct that the hip runtime and tools expect.
    std::string bundleName = getBundleGVName(tt);
    Constant *bundleInit =
        ConstantStruct::get(bundleTy, magic, version, offset, cnull);

    GlobalVariable *g = new GlobalVariable(m, bundleTy, /*isConstant=*/true,
                                           GlobalValue::InternalLinkage,
                                           bundleInit, bundleName);
    g->setSection(getBundleSection());
    g->setAlignment(dl.getPrefTypeAlign(g->getType()));

    return g;
  }

  /// Create a global variable that will contain the "handle" to the fat binary.
  /// The handle is the value returned by \@llvm.kit.gpu.register.devcode. The
  /// handle is saved into this global and read from there by the global dtor.
  /// and passed to to \@llvm.kit.gpu.unregister.devcode.
  virtual GlobalVariable *createBundleHandleGV(Module &m) {
    LLVMContext &ctx = m.getContext();
    std::string name = getBundleHandleGVName(tt);
    PointerType *type = PointerType::getUnqual(ctx);

    Constant *cnull = ConstantPointerNull::get(type);

    GlobalVariable *g = new GlobalVariable(m, type, /*isConstant=*/false,
                                           GlobalValue::InternalLinkage,
                                           /*init=*/cnull, name);
    g->setAlignment(m.getDataLayout().getPointerABIAlignment(0));
    g->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

    return g;
  }

  /// Register non-constant global variables that are present in the device
  /// module, \p devM.
  virtual void registerNonConstGlobals(IRBuilder<> &builder,
                                       Value *bundleHandle,
                                       const Module &devM) {
    Module *m = getModule(builder);
    assert(m && "Builder must be set to a basic block in a module");

    LLVMContext &ctx = m->getContext();
    const DataLayout &dl = m->getDataLayout();

    Constant *ctt = toConstant(tt, ctx);

    // Register any non-constant global variables used in the kernel module.
    // Each of these should have a corresponding global in the host.
    for (const GlobalVariable &devG : devM.globals()) {
      if (devG.isConstant())
        continue;

      GlobalVariable *hostG = m->getGlobalVariable(devG.getName(),
                                                   /*AllowInternal=*/true);
      assert(hostG && "Could not find corresponding global on host");

      uint64_t size = dl.getTypeAllocSize(hostG->getValueType());

      GlobalVariable *gName = createConstString(hostG->getName(), *m);
      Constant *gSize = toConstant(size, ctx);
      Constant *gConst = toConstant(uint32_t(hostG->isConstant()), ctx);
      // FIXME?: Why is this always set to zero? The API is asking if this is
      // "external". Is this asking if it has external linkage? Or is it asking
      // if this is externally defined (as in C's extern)? In either case, why
      // are we always passing 0 here? Is this just the "safer" course, or is it
      // that we just haven't yet encountered a situation where this should be
      // non-zero? Or does AMD require this to be zero currently because it is
      // they who have not implemented something?
      Constant *gExt = toConstant(0U, ctx);

      LLVM_DEBUG(dbgs() << "\t\t\tregister global '" << hostG->getName()
                        << "' via ctor runtime call.\n");
      builder.CreateIntrinsic(
          Intrinsic::kit_gpu_register_global,
          {ctt, bundleHandle, hostG, gName, gName, gSize, gExt, gConst});
    }
  }

  /// Add additional code to the ctor before the device code is registered. The
  /// default genCtor implementation does a lot of the work that is common to
  /// the GPU-centric tapir targets. But some targets may have to add custom
  /// code to the ctor. This callback affords them the chance to do so. This is
  /// called after the common work is done, but before the device code is
  /// registered. Essentially, the structure of the ctor is as shown below:
  ///
  ///     common-code
  ///     genCtorBeforeDevCodeRegistration()
  ///     genCtorDevCodeRegistration()
  ///     genCtorAfterDevCodeRegistration()
  ///
  virtual void genCtorBeforeDevCodeRegistration(IRBuilder<> &builder) {
    // The default implementation does nothing.
  }

  /// Register the device code, and all non-const global variables in the device
  /// code.
  virtual void genCtorDevCodeRegistration(IRBuilder<> &builder,
                                          GlobalVariable *gBundle,
                                          GlobalVariable *gBundleHandle,
                                          const Module &devM) {
    Module *m = gBundle->getParent();
    LLVMContext &ctx = m->getContext();
    Align align = m->getDataLayout().getPointerABIAlignment(0);

    Constant *ctt = toConstant(tt, ctx);

    Value *handle = builder.CreateIntrinsic(Intrinsic::kit_gpu_register_devcode,
                                            {ctt, gBundle});
    builder.CreateAlignedStore(handle, gBundleHandle, align);

    registerNonConstGlobals(builder, handle, devM);
  }

  /// Add additional code to the ctor after the device code and non-const
  /// globals have been registered. Essentially, the structure of the ctor is as
  /// shown below:
  ///
  ///     common-code
  ///     genCtorBeforeDevCodeRegistration()
  ///     genCtorDevCodeRegistration()
  ///     genCtorAfterDevCodeRegistration()
  ///
  virtual void genCtorAfterDevCodeRegistration(IRBuilder<> &builder,
                                               GlobalVariable *gBundleHandle,
                                               const Module &devM) {
    // The default implementation does nothing.
  }

  /// Get the magic number present in the bundle containing the device code.
  virtual int getBundleMagic() const = 0;

  /// Get the version of the bundle.
  virtual int getBundleVersion() const = 0;

  /// Get the object-file section in which the bundle must be present.
  virtual StringRef getBundleSection() const = 0;

public:
  virtual ~PopulateCtorGPU() = default;

  /// Populate the ctor. \p builder must point to the correct insertion point in
  /// the ctor that must have been added to the module.
  void run(IRBuilder<> &builder) {
    Module *mPtr = getModule(builder);
    assert(mPtr && "Insert point of builder is in module");

    Module &m = *mPtr;
    GlobalVariable *gFB = getEmbFBGlobal(tt, m);
    assert(gFB && "Could not find global with embedded device code");

    GlobalVariable *gBC = getEmbBCGlobal(tt, m);
    assert(gBC && "Could not find global with embedded bitcode");

    Expected<std::unique_ptr<Module>> devMOrErr = parseEmbBCGlobal(*gBC);
    if (not devMOrErr)
      exitOnError(devMOrErr.takeError());

    std::unique_ptr<Module> devM = std::move(devMOrErr.get());
    GlobalVariable *gBundle = createBundleGV(m, gFB);
    GlobalVariable *gBundleHandle = createBundleHandleGV(m);

    genCtorBeforeDevCodeRegistration(builder);
    genCtorDevCodeRegistration(builder, gBundle, gBundleHandle, *devM);
    genCtorAfterDevCodeRegistration(builder, gBundleHandle, *devM);
  }

public:
  static std::string getBundleGVName(TTID tt) {
    std::string buf;
    raw_string_ostream os(buf);

    os << ".kit." << tt << ".bundle";
    os.flush();

    return buf;
  }

  static std::string getBundleHandleGVName(TTID tt) {
    std::string buf;
    raw_string_ostream os(buf);

    os << ".kit." << tt << ".handle";
    os.flush();

    return buf;
  }
};

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
class PopulateCtorCuda : public PopulateCtorGPU {
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
    Module *m = getModule(builder);
    assert(m && "Builder must be set to a basic block in a module");

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
  PopulateCtorCuda(const TTOptions &tto, const GenerateCtorOptions &genCtorOpts)
      : PopulateCtorGPU(TTID::Cuda, tto, genCtorOpts) {}
};

/// Helper class to generate a ctor for kithip (Kitsune's runtime for the hip
/// tapir target).
class PopulateCtorHip : public PopulateCtorGPU {
private:
  static constexpr int magic = 0x48495046;
  static constexpr int version = 1;
  static constexpr const char *section = ".hipFatBinSegment";

private:
  virtual int getBundleMagic() const override { return magic; }

  virtual int getBundleVersion() const override { return version; }

  virtual StringRef getBundleSection() const override { return section; }

  virtual void genCtorBeforeDevCodeRegistration(IRBuilder<> &builder) override {
    assert(getModule(builder) &&
           "Builder must be set to a basic block in a module");

    if (tto.getHipXnack() == MaybeBool::On) {
      LLVM_DEBUG(dbgs() << "\t\tenable xnack via ctor runtime call.\n");
      createCall(builder, KitFunc::kithip_enable_xnack);
    }

    if (genCtorOpts.useYLaunch) {
      LLVM_DEBUG(
          dbgs()
          << "\t\tenable y-axis launch pattern via ctor runtime call.\n");
      createCall(builder, KitFunc::kithip_enable_y_axis_launches);
    }
  }

public:
  PopulateCtorHip(const TTOptions &tto, const GenerateCtorOptions &genCtorOpts)
      : PopulateCtorGPU(TTID::Hip, tto, genCtorOpts) {}
};

class GenerateCtorsImpl {
private:
  const TTOptions &tto;
  const GenerateCtorOptions &genCtorOpts;

private:
  static constexpr const char *initOptsGVName = ".kitrt.init.opts";

private:
  // Generate a skeleton for a ctor/dtor. \p name is the name of the function
  // to generate. The returned function will contain exactly two basic blocks.
  // The first will be the entry block of the function. This will not contain
  // any instructions except the terminator, which will be an unconditional
  // branch to the exit block. The exit block will contain a single return
  // instruction (that returns void).
  Function *genFunc(Module &m, StringRef name) {
    LLVMContext &ctx = m.getContext();

    Type *ret = Type::getVoidTy(ctx);
    FunctionType *fty = FunctionType::get(ret, {}, /*IsVarArg=*/false);
    Function *f = Function::Create(fty, GlobalValue::InternalLinkage, name, &m);

    BasicBlock *bbEntry = BasicBlock::Create(ctx, "entry", f);
    BasicBlock *bbExit = BasicBlock::Create(ctx, "exit", f);

    BranchInst::Create(bbExit, bbEntry);
    ReturnInst::Create(ctx, bbExit);

    return f;
  }

  // Generate the global variable containing the runtime configuration. This
  // will be passed to __kitrt_initialize from the ctor.
  GlobalVariable *genInitOptsGV(Module &m) {
    auto isCalled = [](KitFunc libFunc, Module &m) -> bool {
      if (Function *f = getDeclarationIfExists(m, libFunc))
        for (User *user : f->users())
          if (const auto *call = dyn_cast<CallInst>(user))
            if (call->getCalledFunction() == f)
              return true;
      return false;
    };

    // Get the RTID of the runtime corresponding to the given TTID. Not all
    // TTID's have associated runtimes. If the TTID does not, return
    // std::nullopt.
    auto getRTIDFor = [](TTID tt) -> std::optional<kitrt::RTID> {
      switch (tt) {
      case TTID::Cuda: return RT_CUDA;
      case TTID::Hip: return RT_HIP;
      case TTID::OpenCilk: return RT_OPENCILK;
      case TTID::OpenMP: return RT_OPENMP;
      case TTID::Pthreads: return RT_PTHREADS;
      case TTID::Qthreads: return RT_QTHREADS;
      case TTID::Custom:
      case TTID::Serial: return std::nullopt;
      case TTID::Nolo:
        llvm_unreachable("getRTIDFor: Cannot get RTID for TTID::Nolo");
      case TTID::Lambda:
      case TTID::OMPTask:
      case TTID::Realm:
        // These tapir targets are not yet fully supported.
        break;
      }
      llvm_unreachable("getRTIDFor: TTID not handled");
    };

    const SmallSet<TTID, 0> tts = *getTTsAttr(m);
    kitrt::InitOptions initOpts{0};
    for (TTID tt : tts)
      if (std::optional<kitrt::RTID> rt = getRTIDFor(tt))
        initOpts.rts |= static_cast<uint64_t>(*rt);

    // Only enable the PAPI runtime if PAPI instrumentation was recorded.
    if (isCalled(KitFunc::kitpapi_start, m))
      initOpts.rts |= RT_PAPI;

    // Only enable the timing runtime if a timer was started
    if (isCalled(KitFunc::kittimer_start, m))
      initOpts.rts |= RT_TIMER;

    LLVMContext &ctx = m.getContext();
    Constant *initOptsV = toConstant(initOpts, ctx);
    Type *initOptsTy = initOptsV->getType();
    GlobalVariable *initOptsG = new GlobalVariable(
        m, initOptsTy, /*isConstant=*/true, GlobalValue::InternalLinkage,
        /*init=*/initOptsV, initOptsGVName);

    initOptsG->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
    initOptsG->setSection(object::kitSectRTInitOpts);

    return initOptsG;
  }

  // Generate the ctor. This will first initialize Kitsune's runtime by calling
  // __kitrt_initialize. This will be followed any tapir-target-specific
  // configuration that must be carried out.
  void genCtor(Module &m) {
    Function *ctor = genFunc(m, ".kit.ctor");
    IRBuilder<> builder(ctor->getEntryBlock().getTerminator());

    // The main runtime must be initialized before we do anything else.
    GlobalVariable *initOpts = genInitOptsGV(m);
    createCall(builder, KitFunc::kitrt_initialize, {initOpts});

    // Now do anything that needs to be done for the tapir targets.
    const SmallSet<TTID, 0> tts = *getTTsAttr(m);
    for (TTID tt : tts) {
      switch (tt) {
      case TTID::Cuda: //
        PopulateCtorCuda(tto, genCtorOpts).run(builder);
        break;
      case TTID::Hip: //
        PopulateCtorHip(tto, genCtorOpts).run(builder);
        break;
      case TTID::Custom:
      case TTID::OpenCilk:
      case TTID::OpenMP:
      case TTID::Pthreads:
      case TTID::Qthreads:
      case TTID::Serial:
        // Nothing to be done for these tapir targets.
        break;
      case TTID::Nolo:
        llvm_unreachable("Must not generate ctor for nolo tapir target");
        break;
      case TTID::Lambda:
      case TTID::OMPTask:
      case TTID::Realm:
        // These tapir targets are not fully supported yet. Just fall through to
        // the default because the effect is the same.
      default: llvm_unreachable("genCtor: TTID not handled");
      }
    }

    // Do this at the end, otherwise it gets added before the global variable
    // bundle and handle which breaks some tests. The tests probably ought to be
    // written so they don't fail because of things like this, but this is the
    // easier path for now.
    appendToGlobalCtors(m, ctor, kitCtorPriority);
  }

  void populateDtorGPU(TTID tt, IRBuilder<> &builder) {
    Module *mPtr = getModule(builder);
    assert(mPtr && "Insert point of builder is in module");

    Module &m = *mPtr;
    LLVMContext &ctx = m.getContext();
    const DataLayout &dl = m.getDataLayout();

    Align align = dl.getPointerABIAlignment(KitAS::Default);
    PointerType *ptr = PointerType::getUnqual(ctx);

    Constant *ctt = toConstant(tt, ctx);
    std::string name = PopulateCtorGPU::getBundleHandleGVName(tt);
    GlobalVariable *gHandle = m.getGlobalVariable(name, /*allowInternal=*/true);
    Value *handle = builder.CreateAlignedLoad(ptr, gHandle, align);

    builder.CreateIntrinsic(Intrinsic::kit_gpu_unregister_devcode,
                            {ctt, handle});
  }

  void genDtor(Module &m) {
    Function *dtor = genFunc(m, ".kit.dtor");
    IRBuilder<> builder(dtor->getEntryBlock().getTerminator());

    // Do things in the reverse order from the ctor. First, process any tapir
    // targets that need processing.
    SmallSet<TTID, 0> tts = *getTTsAttr(m);
    for (TTID tt : tts) {
      switch (tt) {
      case TTID::Cuda:
      case TTID::Hip: //
        populateDtorGPU(tt, builder);
        break;
      case TTID::Custom:
      case TTID::OpenCilk:
      case TTID::OpenMP:
      case TTID::Pthreads:
      case TTID::Qthreads:
      case TTID::Serial:
        // Nothing to be done for these tapir targets.
        break;
      case TTID::Nolo:
        llvm_unreachable("Must not generate dtor for nolo tapir target");
        break;
      case TTID::Lambda:
      case TTID::OMPTask:
      case TTID::Realm:
        // These tapir targets are not fully supported yet. Just fall through
        // because the effect is the same.
      default: llvm_unreachable("genDtor: TTID not handled");
      }
    }

    // This must be the last thing that this function does.
    GlobalVariable *g =
        m.getGlobalVariable(initOptsGVName, /*AllowInternal=*/true);
    createCall(builder, KitFunc::kitrt_finalize, {g});

    // Do this at the end just to mirror when the ctor priority is set. Doing it
    // earlier doesn't affect any tests.
    appendToGlobalDtors(m, dtor, kitDtorPriority);
  }

public:
  GenerateCtorsImpl(const TTOptions &tto,
                    const GenerateCtorOptions &genCtorOpts)
      : tto(tto), genCtorOpts(genCtorOpts) {}

  void run(Module &m) {
    genCtor(m);
    genDtor(m);
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

  if (!hasTTsAttr(m)) {
    emitDiagnostic(DiagID::WarnGeneric,
                   "kit.module.tts attribute not found in module. No ctors "
                   "will be generated");
    return PreservedAnalyses::all();
  }

  const TTOptions &ttOpts = ttObjs.getOptions();
  GenerateCtorOptions genCtorOpts;
  if (&clUseYLaunch)
    genCtorOpts.useYLaunch = clUseYLaunch;

  GenerateCtorsImpl(ttOpts, genCtorOpts).run(m);

  // This never invalidates any analyses since, at most, only the initializer of
  // a global variable will have changed. The generated ctors will not be called
  // explicitly in the code, so the callgraph will not have changed either.
  return PreservedAnalyses::all();
}
