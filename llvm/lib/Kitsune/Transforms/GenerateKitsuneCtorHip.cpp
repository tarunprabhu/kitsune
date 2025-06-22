//=- GenerateKitsuneCtorHip.cpp - ctor for Kitsune's hip runtime -------*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generate global constructors for Kitsune's hip runtime
//
//===----------------------------------------------------------------------===//

#include "GenerateKitsuneCtorsImpl.h"
#include "kitsune/Analysis/TapirTargetAnalysis.h"
#include "kitsune/Config/config.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/Tapir.h"
#include "kitsune/Core/TapirTargetOptions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

static cl::opt<bool>
    clUseYLaunch("hipabi-y-launch", cl::init(false), cl::Hidden,
                 cl::desc("Launch kernel using y-axis threading."));

namespace {

/// Helper class to generate a ctor for kithip (Kitsune's runtime for the hip
/// tapir target).
class GenerateKitsuneCtorHip {
private:
  static constexpr int magic = 0x48495046;
  static constexpr int version = 1;
  static constexpr const char *section = ".hipFatBinSegment";

private:
  const TapirTargetOptions &tto;
  detail::GetTLI getTLI;

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

    // TODO: Do we really need the ConstantExpr here or can we just pass the
    // global variable directly?
    Constant *zero = ConstantInt::get(idxTy, 0);
    Constant *zeros[] = {zero, zero};

    // Wrap the fatbinary in a struct that the hip runtime and tools expect.
    Constant *bundleInit = ConstantStruct::get(
        bundleTy, ConstantInt::get(i32Ty, magic),
        ConstantInt::get(i32Ty, version),
        ConstantExpr::getGetElementPtr(fatBin->getValueType(), fatBin, zeros),
        ConstantPointerNull::get(ptrTy));

    GlobalVariable *g = new GlobalVariable(m, bundleTy, /*isConstant*/ true,
                                           GlobalValue::InternalLinkage,
                                           bundleInit, ".kithip.bundle");
    g->setSection(section);
    g->setAlignment(dl.getPrefTypeAlign(g->getType()));

    return g;
  }

  /// Create a global variable that will contain the "handle" to the fat binary.
  /// The handle is the value returned by __hipRegisterFatBinary(). The handle
  /// is saved into this global and read from there by the global dtor and
  /// passed to __hipUnregisterFatBinary().
  GlobalVariable *createBundleHandleGV(Module &m) {
    const DataLayout &dl = m.getDataLayout();

    LLVMContext &ctx = m.getContext();
    PointerType *ptrTy = PointerType::getUnqual(ctx);

    GlobalVariable *g = new GlobalVariable(
        m, ptrTy, /*isConstant*/ false, GlobalValue::InternalLinkage,
        ConstantPointerNull::get(ptrTy), ".kithip.handle");
    g->setAlignment(dl.getPointerABIAlignment(0));
    g->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

    return g;
  }

  Function *createCtor(Module &m, const Module &devM, Function *dtor,
                       GlobalVariable *gBundle, GlobalVariable *gBundleHandle) {
    const DataLayout &dl = m.getDataLayout();
    Align alignPtr = dl.getPointerABIAlignment(0);

    LLVMContext &ctx = m.getContext();

    Type *voidTy = Type::getVoidTy(ctx);
    PointerType *ptrTy = PointerType::getUnqual(ctx);
    Type *i32Ty = Type::getInt32Ty(ctx);
    Type *boolTy = Type::getInt8Ty(ctx);

    ConstantInt *constTT = createConstInt(TTID::Hip, ctx);
    Constant *czero = ConstantInt::get(i32Ty, 0);

    FunctionType *ctorTy = FunctionType::get(voidTy, ptrTy, false);
    Function *ctor = Function::Create(ctorTy, GlobalValue::InternalLinkage,
                                      ".kithip.ctor", &m);

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

    if (tto.getHipXnack() == MaybeBool::On)
      LLVM_DEBUG(dbgs() << "\t\tenable xnack via ctor runtime call.\n");

    FunctionCallee kitrtEnableXnack = Intrinsic::getOrInsertDeclaration(
        &m, Intrinsic::kitrt_hip_enable_xnack);
    builder.CreateCall(
        kitrtEnableXnack,
        {ConstantInt::get(boolTy, tto.getHipXnack() == MaybeBool::On)});

    if (clUseYLaunch)
      LLVM_DEBUG(
          dbgs()
          << "\t\tenable y-axis launch pattern via ctor runtime call.\n");
    FunctionCallee kitrtEnableYAxisLaunch = Intrinsic::getOrInsertDeclaration(
        &m, Intrinsic::kitrt_enable_y_axis_launches);
    builder.CreateCall(kitrtEnableYAxisLaunch,
                       {constTT, ConstantInt::get(boolTy, clUseYLaunch)});

    if (unsigned fixedTPB = tto.getFixedThreadsPerBlock()) {
      FunctionCallee kitrtSetFixedTPB =
          Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kitrt_set_fixed_tpb);
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

    FunctionCallee hipRegisterFatBinary =
        getOrInsertLibFunc(&m, tli, LibFunc_hip_register_fat_binary);
    Value *bundleHandle = builder.CreateCall(hipRegisterFatBinary, gBundle);
    builder.CreateAlignedStore(bundleHandle, gBundleHandle, alignPtr);

    // Register any non-constant global variables used in the kernel module.
    // Each of these should have a corresponding global in the host.
    FunctionCallee hipRegisterVar =
        getOrInsertLibFunc(&m, tli, LibFunc_hip_register_var);
    for (const GlobalVariable &devG : devM.globals()) {
      if (devG.isConstant())
        continue;

      GlobalVariable *hostG = m.getGlobalVariable(devG.getName());
      assert(hostG && "Could not find corresponding global on host");

      uint64_t size = dl.getTypeAllocSize(hostG->getType());

      GlobalVariable *gName = createConstString(hostG->getName(), m);
      Constant *gSize = ConstantInt::get(sizeTTy, size);
      Constant *gConst = ConstantInt::get(i32Ty, hostG->isConstant());
      // FIXME?: Why is this always set to zero? The API is asking if this is
      // "external". Is this asking if it has external linkage? Or is it asking
      // if this is externally defined (as in C's extern)? In either case, why
      // are we always passing 0 here? Is this just the "safer" course, or is it
      // that we just haven't yet encountered a situation where this should be
      // non-zero? Or does hip require this to be zero currently because it is
      // they who have not implemented something?
      Constant *gExt = ConstantInt::get(i32Ty, 0);

      Value *args[] = {bundleHandle, hostG, gName, gName, gExt, gSize, gConst,
                       // Per the documentation, The last argument to
                       // hipRegisterVar() must always be zero
                       czero};

      LLVM_DEBUG(dbgs() << "\t\t\tregister global '" << hostG->getName()
                        << "' via ctor runtime call.\n");
      builder.CreateCall(hipRegisterVar, args);
    }

    // Now add the dtor to help us clean up at program exit.
    FunctionCallee atExit = getOrInsertLibFunc(&m, tli, LibFunc_atexit);
    builder.CreateCall(atExit, dtor);

    builder.CreateRetVoid();
    return ctor;
  }

  Function *createDtor(Module &m, GlobalVariable *gBundleHandle) {
    const DataLayout &dl = m.getDataLayout();
    Align alignPtr = dl.getPointerABIAlignment(0);

    LLVMContext &ctx = m.getContext();
    Type *voidTy = Type::getVoidTy(ctx);
    PointerType *ptrTy = PointerType::getUnqual(ctx);

    FunctionType *dtorTy = FunctionType::get(voidTy, ptrTy, false);
    Function *dtor = Function::Create(dtorTy, GlobalValue::InternalLinkage,
                                      ".kithip.dtor", &m);

    TargetLibraryInfo &tli = getTLI(*dtor);
    IRBuilder<> builder(BasicBlock::Create(ctx, "entry", dtor));
    Value *handle = builder.CreateAlignedLoad(ptrTy, gBundleHandle, alignPtr);

    FunctionCallee hipUnregisterFatBinary =
        getOrInsertLibFunc(&m, tli, LibFunc_hip_unregister_fat_binary);
    builder.CreateCall(hipUnregisterFatBinary, handle);

    // FIXME: There is a bug here which seems to cause use-after-free errors in
    // Kitsune's runtime. It is not entirely clear where exactly the problem is.
    // This causes the kitsune-test-suite to consistently fail. In the interest
    // of having the test suite actually be useful, don't generate the call to
    // finalize the runtime until we can figure out exactly what is going on
    // there.
    // ConstantInt *constTT = createConstInt(TTID::Hip, ctx);
    // FunctionCallee kitrtFinalize =
    //     Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kitrt_finalize);
    // builder.CreateCall(kitrtFinalize, {constTT});

    builder.CreateRetVoid();
    return dtor;
  }

public:
  GenerateKitsuneCtorHip(const TapirTargetOptions &tto, detail::GetTLI getTLI)
      : tto(tto), getTLI(getTLI) {}

  void run(Module &m) {
    GlobalVariable *gFB = getEmbFBGlobal(TTID::Hip, m);
    assert(gFB && "Could not find global with embedded hip fat binary");

    GlobalVariable *gBC = getEmbBCGlobal(TTID::Hip, m);
    assert(gBC && "Could not find global with embedded bitcode");

    std::unique_ptr<Module> devM = parseEmbBCGlobal(*gBC);

    GlobalVariable *gBundle = createBundleGV(m, gFB);
    GlobalVariable *gBundleHandle = createBundleHandleGV(m);
    Function *dtor = createDtor(m, gBundleHandle);
    Function *ctor = createCtor(m, *devM, dtor, gBundle, gBundleHandle);

    // Set the priority of this ctor to be very low so it is one of the last to
    // run.
    appendToGlobalCtors(m, ctor, 65536);
  }
};

} // namespace

void llvm::detail::genKitsuneCtorHip(Module &m, const TapirTargetOptions &tto,
                                     detail::GetTLI getTLI) {
  GenerateKitsuneCtorHip(tto, getTLI).run(m);
}
