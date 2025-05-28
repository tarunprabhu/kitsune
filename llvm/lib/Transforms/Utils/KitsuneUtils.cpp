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
#include "llvm/IR/Constants.h"
#include "llvm/IR/KitsuneMetadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;

ConstantInt *llvm::getConstantInt(LLVMContext &ctxt, TapirTargetID tt) {
  IntegerType *i8 = IntegerType::get(ctxt, 8);
  return ConstantInt::get(i8, int(tt), false);
}

/// Serialize the module to LLVM bitcode. Create a constant byte array with this
/// serialized result and return it.
static Constant *serialize(const Module &m) {
  SmallString<256> bcBuf;
  BitcodeWriter bcWriter(bcBuf);
  bcWriter.writeModule(m);
  bcWriter.writeStrtab();

  // Write a null byte to the end of the serialized buffer because it is
  // required when this is deserialized.
  bcBuf.append("\0");

  LLVMContext &ctx = m.getContext();
  size_t bcSize = bcBuf.size();
  return ConstantDataArray::getRaw(bcBuf, bcSize, Type::getInt8Ty(ctx));
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
  g->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

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
         "Global containing embedded bitcode requires an constant array "
         "initializer");
  assert(cast<ConstantDataArray>(g.getInitializer())
             ->getType()
             ->getElementType()
             ->isIntegerTy(8) &&
         "Global containing embedded bitcode must be a byte array");

  LLVMContext &ctx = g.getContext();
  const Constant *bcInit = g.getInitializer();
  StringRef bcBytes = cast<ConstantDataArray>(bcInit)->getRawDataValues();
  std::unique_ptr<MemoryBuffer> bcBuf = MemoryBuffer::getMemBuffer(bcBytes);
  assert(isBitcode((const unsigned char *)bcBuf->getBufferStart(),
                   (const unsigned char *)bcBuf->getBufferEnd()) &&
         "Global does not contain bitcode");

  Expected<std::unique_ptr<Module>> moduleOrErr = parseBitcodeFile(*bcBuf, ctx);
  if (not moduleOrErr) {
    errs() << moduleOrErr.takeError() << "\n";
    report_fatal_error("Error parsing embedded bitcode");
  }

  return std::move(moduleOrErr.get());
}

GlobalVariable *llvm::createEmbeddedBC(const Module &m, TapirTargetID tt,
                                       Module &hostM) {
  GlobalVariable *g = ::createEmbeddedBC(m, hostM);
  setKitsuneBCMD(*g, tt);

  return g;
}

GlobalVariable *llvm::getEmbeddedBC(TapirTargetID tt, Module &m) {
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

GlobalVariable *llvm::createEmbeddedFB(TapirTargetID tt, Module &m) {
  std::unique_ptr<MemoryBuffer> buf = MemoryBuffer::getMemBuffer("");
  GlobalVariable *g = ::createEmbeddedFB(*buf, m);
  setKitsuneFBMD(*g, tt);

  switch (tt) {
  case TapirTargetID::Cuda:
    g->setSection(".nv_fatbin");
    break;
  case TapirTargetID::Hip:
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

GlobalVariable *llvm::getOrCreateGlobalString(Module &m, StringRef s,
                                              StringRef name) {
  for (GlobalVariable &g : m.globals())
    if (g.isConstant() and g.hasInitializer())
      if (auto *cda = dyn_cast<ConstantDataArray>(g.getInitializer()))
        if (cda->getAsCString() == s)
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
