//===- KitsuneUtils.cpp - Helper functions for Kitsune ---------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions for Kitsune-specific utilities.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/KitsuneUtils.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Frontend/Tapir/OptLevelUtils.h"
#include "llvm/Frontend/Tapir/TapirTargetOptions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/KitsuneMetadata.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

StructType *llvm::getKernelInstMixType(LLVMContext &ctx) {
  Type *i64 = Type::getInt64Ty(ctx);
  return StructType::get(i64,  // number of memory ops
                         i64,  // number of floating point ops
                         i64,  // number of integer ops
                         i64); // number of other ops
}

ConstantInt *llvm::getConstantInt(LLVMContext &ctxt, TapirTargetID tt) {
  IntegerType *i8 = IntegerType::get(ctxt, 8);
  return ConstantInt::get(i8, int(tt), false);
}

/// Serialize the module to LLVM bitcode. Create a constant byte array with this
/// serialized result and return it.
static Constant *serialize(const Module &m) {
  SmallString<4096> buf;
  raw_svector_ostream os(buf);
  WriteBitcodeToFile(m, os);

  LLVMContext &ctx = m.getContext();
  return ConstantDataArray::getString(ctx, buf, /*AddNull=*/false);
}

static GlobalVariable *createEmbeddedBC(const Module &m, Module &hostM) {
  // The linkage of the global variable is external to prevent it from being
  // DCE'ed. It must have a name, otherwise a global variable optimization pass
  // will remove it since it will not be used anywhere.
  GlobalValue::LinkageTypes linkage = GlobalValue::ExternalLinkage;
  Constant *bcInit = serialize(m);
  Type *bcType = bcInit->getType();
  GlobalVariable *g = new GlobalVariable(hostM, bcType, /*isConstant=*/true,
                                         linkage, bcInit, ".kitsune.emb.bc");
  g->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

  return g;
}

static GlobalVariable *createEmbeddedFB(MemoryBufferRef buf, Module &m) {
  // The linkage of the global variable is external to prevent it from being
  // DCE'ed.
  GlobalValue::LinkageTypes linkage = GlobalValue::ExternalLinkage;
  LLVMContext &ctx = m.getContext();
  Constant *bcInit =
      ConstantDataArray::getString(ctx, buf.getBuffer(), /*AddNull=*/false);
  Type *bcType = bcInit->getType();
  GlobalVariable *g = new GlobalVariable(m, bcType, /*isConstant=*/true,
                                         linkage, bcInit, ".kitsune.emb.fb");

  return g;
}

std::unique_ptr<Module> llvm::parseEmbeddedBC(const GlobalVariable &g) {
  assert(g.hasInitializer() &&
         "Global containing embedded bitcode requires initializer");
  assert(isa<ConstantDataArray>(g.getInitializer()) &&
         "Global containing embedded bitcode requires a constant array "
         "initializer");
  assert(cast<ConstantDataArray>(g.getInitializer())
             ->getType()
             ->getElementType()
             ->isIntegerTy(8) &&
         "Global containing embedded bitcode must be a byte array");

  LLVMContext &ctx = g.getContext();
  const Constant *bcInit = g.getInitializer();
  StringRef bcBytes = cast<ConstantDataArray>(bcInit)->getAsString();
  std::unique_ptr<MemoryBuffer> bcBuf = MemoryBuffer::getMemBuffer(bcBytes);
  assert(isBitcode((const unsigned char *)bcBuf->getBufferStart(),
                   (const unsigned char *)bcBuf->getBufferEnd()) &&
         "Global does not contain bitcode");

  Expected<std::unique_ptr<Module>> moduleOrErr = parseBitcodeFile(*bcBuf, ctx);
  if (not moduleOrErr) {
    Error err = moduleOrErr.takeError();
    handleAllErrors(std::move(err), [&](ErrorInfoBase &e) {
      errs() << "Error parsing embedded bitcode: " << e.message() << "\n";
    });
    report_fatal_error("Could not parse embedded bitcode");
  }

  std::unique_ptr<Module> m = std::move(moduleOrErr.get());
  if (std::optional<StringRef> name = getNameFromModuleMD(*m))
    m->setModuleIdentifier(*name);

  return m;
}

GlobalVariable *llvm::createEmbeddedBC(const Module &m, TapirTargetID tt,
                                       Module &hostM) {
  GlobalVariable *g = ::createEmbeddedBC(m, hostM);
  setKitsuneBCMD(*g, tt);

  return g;
}

GlobalVariable *llvm::getEmbeddedBC(TapirTargetID tt, Module &m) {
  // This assumes that only a single embedded bitcode module exists for a given
  // tapir target. This is the current implementation and might change, though
  // that is unlikely.
  for (GlobalVariable &g : m.globals())
    if (hasKitsuneBCMD(g, tt))
      return &g;
  return nullptr;
}

GlobalVariable *llvm::resetEmbeddedBC(const Module &m, GlobalVariable &g) {
  assert(g.getParent() && "Global with embedded bitcode must be in a module");

  Module &hostM = *g.getParent();
  GlobalVariable *newG = ::createEmbeddedBC(m, hostM);

  if (g.hasSection())
    newG->setSection(g.getSection());
  newG->copyMetadata(&g, 0);
  g.replaceAllUsesWith(newG);
  g.eraseFromParent();

  return newG;
}

std::unique_ptr<Module> llvm::getEmbeddedModule(TapirTargetID tt, Module &m) {
  if (GlobalVariable *g = getEmbeddedBC(tt, m))
    return parseEmbeddedBC(*g);
  return nullptr;
}

EmbeddedModulesMapTy llvm::getEmbeddedModules(Module &m) {
  EmbeddedModulesMapTy embBCs;
  for (TapirTargetID tt : getTargetsGeneratingEmbBC())
    if (const GlobalVariable *g = getEmbeddedBC(tt, m))
      embBCs.emplace(tt, parseEmbeddedBC(*g));
  return embBCs;
}

GlobalVariable *llvm::createEmbeddedFB(TapirTargetID tt, Module &m) {
  std::unique_ptr<MemoryBuffer> buf = MemoryBuffer::getMemBuffer("");
  GlobalVariable *g = ::createEmbeddedFB(*buf, m);
  setKitsuneFBMD(*g, tt);

  switch (tt) {
  case TapirTargetID::Cuda:
    g->setSection(".nv_fatbin");
    break;
  case TapirTargetID::Hip:
    g->setSection(".hip_fatbin");
    g->setAlignment(Align(4096));
    break;
  default:
    llvm_unreachable(
        "Creating embedded fat binary with unexpected tapir target");
    break;
  }
  return g;
}

GlobalVariable *llvm::getEmbeddedFB(TapirTargetID tt, Module &m) {
  for (GlobalVariable &g : m.globals())
    if (hasKitsuneFBMD(g, tt))
      return &g;
  return nullptr;
}

GlobalVariable *llvm::resetEmbeddedFB(MemoryBufferRef buf, GlobalVariable &g) {
  assert(g.getParent() && "Global with embedded bitcode must be in a module");

  Module &m = *g.getParent();
  GlobalVariable *newG = ::createEmbeddedFB(buf, m);

  if (g.hasSection())
    newG->setSection(g.getSection());
  newG->copyMetadata(&g, 0);
  g.replaceAllUsesWith(newG);
  g.eraseFromParent();

  return newG;
}

GlobalVariable *llvm::createKernelMDGlobal(Module &m, StringRef kname) {
  LLVMContext &ctx = m.getContext();
  StructType *type = getKernelInstMixType(ctx);
  Constant *init = Constant::getNullValue(type);
  auto *g = new GlobalVariable(m, type, /*IsConstant=*/true,
                               GlobalValue::PrivateLinkage, init);
  g->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  setKitsuneKernelMDMD(*g, kname);

  return g;
}

GlobalVariable *llvm::getOrCreateGlobalString(Module &m, StringRef s,
                                              StringRef name) {
  for (GlobalVariable &g : m.globals())
    if (g.isConstant() and g.hasInitializer())
      if (auto *cda = dyn_cast<ConstantDataArray>(g.getInitializer()))
        if (cda->isCString() and cda->getAsCString() == s)
          return &g;

  LLVMContext &ctx = m.getContext();
  Constant *init = ConstantDataArray::getString(ctx, s, true);
  Type *type = init->getType();
  auto *g = new GlobalVariable(m, type, /*IsConstant=*/true,
                               GlobalValue::PrivateLinkage, init, name);
  g->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  g->setAlignment(Align(1));
  return g;
}

static TargetMachine *createAMDGPUTargetMachine(const TapirTargetOptions &tto) {
  Triple triple("amdgcn", "amd", "amdhsa");

  std::string err;
  const Target *target = TargetRegistry::lookupTarget("", triple, err);
  assert(target && "Unable to find registered AMDGPU target");

  // TODO: Should we allow relocations here?
  CodeModel::Model codeModel = CodeModel::Small;
  Reloc::Model relocModel = Reloc::Static;
  CodeGenOptLevel optLevel = mapToCodeGenOptLevel(tto.getOptLevel());
  TargetOptions opts;
  opts.UseInitArray = true;
  opts.EmitAddrsig = true;
  opts.AllowFPOpFusion = tto.getFPOpFusionMode();

  return target->createTargetMachine(triple.str(), tto.getHipArch(),
                                     tto.getHipTargetFeatures(), opts,
                                     relocModel, codeModel, optLevel);
}

static TargetMachine *createPTXTargetMachine(const TapirTargetOptions &tto) {
  Triple triple("nvptx64", "nvidia", "cuda");

  std::string err;
  const Target *target = TargetRegistry::lookupTarget("", triple, err);
  assert(target && "Unable to find registered PTX target");

  CodeModel::Model codeModel = CodeModel::Small;
  Reloc::Model relocModel = Reloc::PIC_;
  CodeGenOptLevel optLevel = mapToCodeGenOptLevel(tto.getOptLevel());
  TargetOptions opts;
  opts.AllowFPOpFusion = tto.getFPOpFusionMode();

  return target->createTargetMachine(triple.str(), tto.getCudaArch(),
                                     tto.getCudaTargetFeatures(), opts,
                                     relocModel, codeModel, optLevel);
}

TargetMachine *llvm::createTargetMachine(TapirTargetID tt,
                                         const TapirTargetOptions &tto) {
  switch (tt) {
  case TapirTargetID::Cuda:
    return createPTXTargetMachine(tto);
  case TapirTargetID::Hip:
    return createAMDGPUTargetMachine(tto);
  default:
    llvm_unreachable("createTargetMachine: TapirTargetID not handled");
  }
}

constexpr std::array<TapirTargetID, 2> tgtsGenEmbBC = {TapirTargetID::Cuda,
                                                       TapirTargetID::Hip};
const std::array<TapirTargetID, 2> &llvm::getTargetsGeneratingEmbBC() {
  return tgtsGenEmbBC;
}

bool llvm::generatesEmbBC(TapirTargetID tt) {
  switch (tt) {
  case TapirTargetID::Cuda:
  case TapirTargetID::Hip:
    return true;
  default:
    return false;
  }
}
