//===- EmbUtils.cpp - Helper functions for embedded data in modules -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of utilities to create, retrieve and modify embedded data in
// an LLVM Module.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/TargetUtils.h"
#include "kitsune/Support/TTUtils.h"
#include "kitsune/Support/ToString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

/// Serialize the module to LLVM bitcode. Create a constant byte array with this
/// serialized result and return it.
static Constant *serialize(const Module &m) {
  SmallString<4096> buf;
  raw_svector_ostream os(buf);
  WriteBitcodeToFile(m, os);

  LLVMContext &ctx = m.getContext();
  return ConstantDataArray::getString(ctx, buf, /*AddNull=*/false);
}

static GlobalVariable *createEmbBCGlobal(const Module &m, Module &hostM) {
  // The linkage of the global variable is external to prevent it from being
  // DCE'ed. It must have a name, otherwise a global variable optimization pass
  // will remove it since it will not be used anywhere.
  GlobalValue::LinkageTypes linkage = GlobalValue::ExternalLinkage;
  Constant *init = serialize(m);
  Type *type = init->getType();
  GlobalVariable *g = new GlobalVariable(hostM, type, /*isConstant=*/true,
                                         linkage, init, ".kitsune.emb.bc");
  g->addAttribute(Attribute::KitBC);
  g->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

  return g;
}

static GlobalVariable *createEmbFBGlobal(MemoryBufferRef buf, Module &m) {
  // The linkage of the global variable is external to prevent it from being
  // DCE'ed.
  GlobalValue::LinkageTypes linkage = GlobalValue::ExternalLinkage;
  StringRef data = buf.getBuffer();
  LLVMContext &ctx = m.getContext();
  Constant *init = ConstantDataArray::getString(ctx, data, /*AddNull=*/false);
  Type *type = init->getType();
  GlobalVariable *g = new GlobalVariable(m, type, /*isConstant=*/true, linkage,
                                         init, ".kitsune.emb.fb");
  g->addAttribute(Attribute::KitFB);

  return g;
}

std::unique_ptr<Module> llvm::parseEmbBCGlobal(const GlobalVariable &g) {
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
  const Constant *init = g.getInitializer();
  StringRef bytes = cast<ConstantDataArray>(init)->getAsString();
  std::unique_ptr<MemoryBuffer> buf = MemoryBuffer::getMemBuffer(bytes);
  assert(isBitcode((const unsigned char *)buf->getBufferStart(),
                   (const unsigned char *)buf->getBufferEnd()) &&
         "Global does not contain bitcode");

  Expected<std::unique_ptr<Module>> moduleOrErr = parseBitcodeFile(*buf, ctx);
  if (not moduleOrErr) {
    Error err = moduleOrErr.takeError();
    handleAllErrors(std::move(err), [&](ErrorInfoBase &e) {
      errs() << "Error parsing embedded bitcode: " << e.message() << "\n";
    });
    report_fatal_error("Could not parse embedded bitcode");
  }

  // When a module is serialized, its identifier is not recorded in the bitcode.
  // We add kitsune-specific metadata to the module that contains this name so
  // it can be restored when it is deserialized.
  std::unique_ptr<Module> m = std::move(moduleOrErr.get());
  if (std::optional<StringRef> name = getNameFromDeviceModuleMetadata(*m))
    m->setModuleIdentifier(*name);

  return m;
}

GlobalVariable *llvm::createEmbBCGlobal(const Module &devM, TTID tt,
                                        Module &hostM) {
  assert(!getEmbBCGlobal(tt, hostM) &&
         "Embedded bitcode global already exists");
  LLVMContext &ctx = hostM.getContext();
  GlobalVariable *g = ::createEmbBCGlobal(devM, hostM);
  g->addAttribute(Attribute::getWithTTID(ctx, tt));

  return g;
}

GlobalVariable *llvm::getEmbBCGlobal(TTID tt, Module &m) {
  // This assumes that only a single embedded bitcode module exists for a given
  // tapir target. This is the current implementation and might change, though
  // that is unlikely.
  for (GlobalVariable &g : m.globals()) {
    if (g.hasAttribute(Attribute::KitBC)) {
      assert(g.hasAttribute(Attribute::KitTT) &&
             "Attribute 'kit_bc' requires 'kit_tt");
      if (g.getAttribute(Attribute::KitTT).getTTID() == tt)
        return &g;
    }
  }
  return nullptr;
}

GlobalVariable *llvm::resetEmbBCGlobal(const Module &devM, GlobalVariable &g) {
  assert(g.getParent() && "Global with embedded bitcode must be in a module");

  Module &hostM = *g.getParent();
  GlobalVariable *newG = ::createEmbBCGlobal(devM, hostM);

  if (g.hasName()) {
    StringRef name = g.getName();
    g.setName("");
    newG->setName(name);
  }
  if (g.hasSection())
    newG->setSection(g.getSection());
  newG->copyAttributesFrom(&g);
  newG->copyMetadata(&g, 0);
  g.replaceAllUsesWith(newG);
  g.eraseFromParent();

  return newG;
}

std::unique_ptr<Module> llvm::getEmbModule(TTID tt, Module &m) {
  if (GlobalVariable *g = getEmbBCGlobal(tt, m))
    return parseEmbBCGlobal(*g);
  return nullptr;
}

EmbModulesMapTy llvm::getEmbModules(const Module &m) {
  EmbModulesMapTy embBCs;
  for (TTID tt : ttsGenEmbBC())
    for (const GlobalVariable &g : m.globals())
      if (g.hasAttribute(Attribute::KitBC) && g.hasAttribute(Attribute::KitTT))
        if (g.getAttribute(Attribute::KitTT).getTTID() == tt)
          embBCs.emplace(tt, parseEmbBCGlobal(g));
  return embBCs;
}

GlobalVariable *llvm::createEmbFBGlobal(TTID tt, Module &hostM) {
  assert(!getEmbFBGlobal(tt, hostM) &&
         "Embedded fat binary global already exists");
  LLVMContext &ctx = hostM.getContext();
  std::unique_ptr<MemoryBuffer> buf = MemoryBuffer::getMemBuffer("");
  GlobalVariable *g = ::createEmbFBGlobal(*buf, hostM);
  g->addAttribute(Attribute::getWithTTID(ctx, tt));

  switch (tt) {
  case TTID::Cuda:
    g->setSection(".nv_fatbin");
    break;
  case TTID::Hip:
    // FIXME: This is temporarily set to experiment with the kithip fatbin
    // globals. Eventually, this should be a name that is unlikely to collide
    // with the corresponding fatbin variables from other translation units.
    g->setName("__kithip_fatbin");
    g->setSection(".hip_fatbin");
    // FIXME: This alignment is not necessary.
    g->setAlignment(Align(4096));
    break;
  default:
    llvm_unreachable(
        "Creating embedded fat binary with unexpected tapir target");
    break;
  }
  return g;
}

GlobalVariable *llvm::getEmbFBGlobal(TTID tt, Module &m) {
  for (GlobalVariable &g : m.globals()) {
    if (g.hasAttribute(Attribute::KitFB)) {
      assert(g.hasAttribute(Attribute::KitTT) &&
             "Attribute 'kit_bc' requires 'kit_tt");
      if (g.getAttribute(Attribute::KitTT).getTTID() == tt)
        return &g;
    }
  }
  return nullptr;
}

GlobalVariable *llvm::resetEmbFBGlobal(MemoryBufferRef buf, GlobalVariable &g) {
  assert(g.getParent() && "Global with embedded bitcode must be in a module");

  Module &m = *g.getParent();
  GlobalVariable *newG = ::createEmbFBGlobal(buf, m);

  if (g.hasName()) {
    StringRef name = g.getName();
    g.setName("");
    newG->setName(name);
  }
  if (g.hasSection())
    newG->setSection(g.getSection());
  newG->copyAttributesFrom(&g);
  newG->copyMetadata(&g, 0);
  g.replaceAllUsesWith(newG);
  g.eraseFromParent();

  return newG;
}

std::unique_ptr<Module> llvm::createEmbModule(TTID tt,
                                              const TapirTargetOptions &tto,
                                              const Module &hostM) {
  LLVMContext &ctx = hostM.getContext();
  TargetMachine *tm = createTargetMachine(tt, tto);

  std::unique_ptr<Module> devM = std::make_unique<Module>("", ctx);

  devM->setTargetTriple(tm->getTargetTriple().str());
  devM->setDataLayout(tm->createDataLayout());
  devM->setModuleIdentifier(join_items("_", "__kit", toString(tt),
                                       sys::path::filename(hostM.getName())));
  addDeviceModuleMetadata(tt, *devM);

  return devM;
}
