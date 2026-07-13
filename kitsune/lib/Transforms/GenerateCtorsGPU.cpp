//===- GenerateCtorsGPU.cpp - Generate ctors for GPU tapir targets --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic implementation of ctor/dtor generation for GPU-centric tapir targets.
//
//===----------------------------------------------------------------------===//

#include "GenerateCtorsGPU.h"
#include "kitsune/Core/BasicBlockUtils.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/TTOptions.h"
#include "kitsune/Support/ErrorHandling.h"
#include "kitsune/Support/OstreamUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

detail::GenerateCtorGPU::GenerateCtorGPU(TTID tt, const TTOptions &tto,
                                         const GenerateCtorOptions &genCtorOpts)
    : GenerateCtorBase(tt, tto), genCtorOpts(genCtorOpts) {}

std::string detail::GenerateCtorGPU::getBundleGVName() {
  std::string buf;
  raw_string_ostream os(buf);

  os << ".kit." << tt << ".bundle";
  os.flush();

  return buf;
}

std::string detail::GenerateCtorGPU::getBundleHandleGVName() {
  std::string buf;
  raw_string_ostream os(buf);

  os << ".kit." << tt << ".handle";
  os.flush();

  return buf;
}

GlobalVariable *
detail::GenerateCtorGPU::createBundleGV(Module &m, GlobalVariable *devCode) {
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
  std::string bundleName = getBundleGVName();
  Constant *bundleInit =
      ConstantStruct::get(bundleTy, magic, version, offset, cnull);

  GlobalVariable *g =
      new GlobalVariable(m, bundleTy, /*isConstant=*/true,
                         GlobalValue::InternalLinkage, bundleInit, bundleName);
  g->setSection(getBundleSection());
  g->setAlignment(dl.getPrefTypeAlign(g->getType()));

  return g;
}

GlobalVariable *detail::GenerateCtorGPU::createBundleHandleGV(Module &m) {
  LLVMContext &ctx = m.getContext();
  std::string name = getBundleHandleGVName();
  PointerType *type = PointerType::getUnqual(ctx);

  Constant *cnull = ConstantPointerNull::get(type);

  GlobalVariable *g = new GlobalVariable(m, type, /*isConstant=*/false,
                                         GlobalValue::InternalLinkage,
                                         /*init=*/cnull, name);
  g->setAlignment(m.getDataLayout().getPointerABIAlignment(0));
  g->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

  return g;
}

void detail::GenerateCtorGPU::registerNonConstGlobals(IRBuilder<> &builder,
                                                      Value *bundleHandle,
                                                      const Module &devM) {
  Module *m = getModule(*builder.GetInsertBlock());
  LLVMContext &ctx = m->getContext();
  const DataLayout &dl = m->getDataLayout();

  Constant *ctt = toConstant(tt, ctx);

  // Register any non-constant global variables used in the kernel module. Each
  // of these should have a corresponding global in the host.
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
    // "external". Is this asking if it has external linkage? Or is it asking if
    // this is externally defined (as in C's extern)? In either case, why are we
    // always passing 0 here? Is this just the "safer" course, or is it that we
    // just haven't yet encountered a situation where this should be non-zero?
    // Or does AMD require this to be zero currently because it is they who have
    // not implemented something?
    Constant *gExt = toConstant(0U, ctx);

    LLVM_DEBUG(dbgs() << "\t\t\tregister global '" << hostG->getName()
                      << "' via ctor runtime call.\n");
    builder.CreateIntrinsic(
        Intrinsic::kit_gpu_register_global,
        {ctt, bundleHandle, hostG, gName, gName, gSize, gExt, gConst});
  }
}

void detail::GenerateCtorGPU::genCtorDevCodeRegistration(
    IRBuilder<> &builder, GlobalVariable *gBundle,
    GlobalVariable *gBundleHandle, const Module &devM) {
  Module *m = gBundle->getParent();
  LLVMContext &ctx = m->getContext();
  Align align = m->getDataLayout().getPointerABIAlignment(0);

  Constant *ctt = toConstant(tt, ctx);

  Value *handle = builder.CreateIntrinsic(Intrinsic::kit_gpu_register_devcode,
                                          {ctt, gBundle});
  builder.CreateAlignedStore(handle, gBundleHandle, align);

  registerNonConstGlobals(builder, handle, devM);
}

Function *detail::GenerateCtorGPU::genCtor(Module &m, GlobalVariable *gBundle,
                                           GlobalVariable *gBundleHandle,
                                           const Module &devM) {
  LLVMContext &ctx = m.getContext();

  // Booleans are always 8-bit integers. toConstant would, otherwise return an
  // i1, but the intrinsic expects i8. Casting the boolean to i8 ensures that we
  // get a value of the correct type.
  Constant *cVerbose = toConstant(uint8_t(tto.getKitrtVerbose()), ctx);
  Constant *ctt = toConstant(tt, ctx);

  Function *ctor = genCtorSkeleton(m);
  IRBuilder<> builder = getBuilderForSkeleton(ctor);

  builder.CreateIntrinsic(Intrinsic::kit_runtime_initialize, ctt);

  // Enable verbose mode early in the constructor so all verbose statements are
  // printed after the runtime has been initialized.
  builder.CreateIntrinsic(Intrinsic::kit_runtime_set_verbose, {ctt, cVerbose});

  // If the MaxThreadsPerBlock has not been set, use a value of 1024 anyway. At
  // the time of writing, exceeding this value degrades performance. This might
  // change, and we may even have to set a different value depending on the
  // specific GPU architecture.
  //
  // FIXME: Don't hardcode this value here. Maybe move it to a named constant.
  unsigned maxTPB = tto.getMaxThreadsPerBlock();
  if (!maxTPB)
    maxTPB = 1024;
  Constant *cTPB = toConstant(maxTPB, ctx);
  builder.CreateIntrinsic(Intrinsic::kit_runtime_set_max_tpb, {ctt, cTPB});

  if (unsigned fixedTPB = tto.getFixedThreadsPerBlock()) {
    Constant *cTPB = toConstant(fixedTPB, ctx);
    builder.CreateIntrinsic(Intrinsic::kit_runtime_set_fixed_tpb, {ctt, cTPB});
  }

  genCtorBeforeDevCodeRegistration(builder);
  genCtorDevCodeRegistration(builder, gBundle, gBundleHandle, devM);
  genCtorAfterDevCodeRegistration(builder, gBundleHandle, devM);

  // We don't need to do anything more because genCtorSkeleton() will have set
  // up dedicated exit blocks and return instructions already.
  return ctor;
}

Function *detail::GenerateCtorGPU::genDtor(Module &m,
                                           GlobalVariable *gBundleHandle) {
  LLVMContext &ctx = m.getContext();
  Align align = m.getDataLayout().getPointerABIAlignment(0);
  PointerType *ptr = PointerType::getUnqual(ctx);

  Constant *ctt = toConstant(tt, ctx);

  Function *dtor = genDtorSkeleton(m);
  IRBuilder<> builder = getBuilderForSkeleton(dtor);

  Value *handle = builder.CreateAlignedLoad(ptr, gBundleHandle, align);
  builder.CreateIntrinsic(Intrinsic::kit_gpu_unregister_devcode, {ctt, handle});
  builder.CreateIntrinsic(Intrinsic::kit_runtime_finalize, ctt);

  // We don't need to do anything more because genCtorSkeleton() will have set
  // up dedicated exit blocks and return instructions already.
  return dtor;
}

void detail::GenerateCtorGPU::run(Module &m) {
  GlobalVariable *gFB = getEmbFBGlobal(tt, m);
  assert(gFB && "Could not find global with embedded cuda fat binary");

  GlobalVariable *gBC = getEmbBCGlobal(tt, m);
  assert(gBC && "Could not find global with embedded bitcode");

  Expected<std::unique_ptr<Module>> devMOrErr = parseEmbBCGlobal(*gBC);
  if (not devMOrErr)
    exitOnError(devMOrErr.takeError());

  std::unique_ptr<Module> devM = std::move(devMOrErr.get());
  GlobalVariable *gBundle = createBundleGV(m, gFB);
  GlobalVariable *gBundleHandle = createBundleHandleGV(m);

  Function *ctor = genCtor(m, gBundle, gBundleHandle, *devM);
  appendToGlobalCtors(m, ctor, kitCtorPriority);

  Function *dtor = genDtor(m, gBundleHandle);
  appendToGlobalDtors(m, dtor, kitDtorPriority);
}
