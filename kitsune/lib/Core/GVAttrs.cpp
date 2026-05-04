//===- GVAttrs.cpp - Kitsune-specific attributes for global variables -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions and utilities to work with Kitsune-specific attributes for global
// variables.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/GVAttrs.h"
#include "AttrsImpl.h"
#include "GVAttrsImpl.h"
#include "VerifierImpl.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/GVUtils.h"
#include "kitsune/Core/ModuleAttrs.h"
#include "kitsune/Core/TypeUtils.h"
#include "kitsune/Core/Verifier.h"
#include "kitsune/Support/ErrorHandling.h"
#include "kitsune/Support/TTIDUtils.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"

using namespace llvm;

//------------------------------------------------------------------------------

MDNode *llvm::detail::getRawAttrList(const GlobalVariable &g) {
  return g.getMetadata(LLVMContext::MD_kit_gv_attrs);
}

void llvm::detail::setAttrList(GlobalVariable &g, MDNode *attrList) {
  g.setMetadata(LLVMContext::MD_kit_gv_attrs, attrList);
}

void llvm::detail::addAttr(GlobalVariable &g, StringRef name,
                           ArrayRef<Metadata *> vals) {
  LLVMContext &ctx = g.getContext();
  MDNode *attrList = getRawAttrList(g);
  MDNode *newAttrList = getAttrListWith(name, vals, attrList, ctx);

  setAttrList(g, newAttrList);
}

void llvm::detail::removeAttr(GlobalVariable &g, StringRef attrName) {
  MDNode *attrList = getRawAttrList(g);
  MDNode *newAttrList = getAttrListWithout(attrName, attrList);

  setAttrList(g, newAttrList);
}

void llvm::detail::verifyAttr(KitVerifier &v, const GlobalVariable &g,
                              StringRef attrName) {
#define GV_ATTR(NAME, IRNAME, ...)                                             \
  if (attrName == IRNAME)                                                      \
    return verify##NAME##Attr(v, g);
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
}

//------------------------------------------------------------------------------

raw_ostream &llvm::operator<<(raw_ostream &os, const GVAttrKind &attr) {
  return os << getAttrName(attr);
}

StringRef llvm::getAttrName(GVAttrKind attr) {
  switch (attr) {
#define GV_ATTR(NAME, IRNAME, ...)                                             \
  case GVAttrKind::NAME:                                                       \
    return IRNAME;
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
  }
  llvm_unreachable("getAttrName: Attribute not handled");
}

std::optional<GVAttrKind> llvm::getGVAttrKind(StringRef name) {
  return StringSwitch<std::optional<GVAttrKind>>(name)
#define GV_ATTR(NAME, IRNAME, ...) .Case(IRNAME, GVAttrKind::NAME)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
      .Default(std::nullopt);
}

void llvm::addAttr(GlobalVariable &g, GVAttrKind attr) {
  switch (attr) {
  default:
    emitDiagnostic(DiagID::ErrAttrAdd, getAttrName(attr));
    exitOnError();
    break;
#define GV_ATTR_0(NAME, IRNAME, ...)                                           \
  case GVAttrKind::NAME:                                                       \
    return detail::addAttr(g, IRNAME, {});
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"
  }
}

DEFN_ATTR_GENERIC(GlobalVariable, GVAttrKind)

#define GV_ATTR(...) DEFN_ATTR_COMMON(GlobalVariable, GVAttrKind, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_0(...) DEFN_ATTR_0(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_1(...) DEFN_ATTR_1(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_2(...) DEFN_ATTR_2(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_3(...) DEFN_ATTR_3(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_4(...) DEFN_ATTR_4(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_5(...) DEFN_ATTR_5(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_6(...) DEFN_ATTR_6(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_7(...) DEFN_ATTR_7(GlobalVariable, __VA_ARGS__)
#define GV_ATTR_8(...) DEFN_ATTR_8(GlobalVariable, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

#define GV_ATTR_N(...) DEFN_ATTR_N(GlobalVariable, __VA_ARGS__)
#define GET_GV_ATTRS
#include "kitsune/Core/GVAttrs.inc"

// -----------------------------------------------------------------------------
// Add custom attribute verifiers here. In general, you should not modify
// anything above this line unless you are modifying a core part of the
// attribute implementation.

static void verifyEmbModuleMetadata(KitVerifier &v, const Module &embM) {
  ModuleAttrKind reqd = ModuleAttrKind::DeviceModuleFlags;
  v.check(hasDeviceModuleFlagsAttr(embM), DiagID::ErrEmbModuleGeneric,
          formatv("missing required attribute '{}'", reqd).str());
}

static void verifyEmbModule(KitVerifier &v, const Module &embM,
                            TTID ttFromHostGV) {
  for (const GlobalVariable &g : embM.globals()) {
    v.check(!hasBitCodeAttr(g), g, DiagID::ErrEmbModuleGeneric,
            "cannot contain embedded bitcode");
    v.check(!hasDeviceCodeAttr(g), g, DiagID::ErrEmbModuleGeneric,
            "cannot contain embedded device code");
  }

  verifyEmbModuleMetadata(v, embM);
  if (std::optional<TTID> tt = getTTIDFromDeviceModuleFlagsAttr(embM))
    v.check(tt == ttFromHostGV, DiagID::ErrEmbModuleGeneric,
            "tapir target in device module flags metadata must match "
            "tapir target in host embedded bitcode global variable");

  v.check(verifyModule(embM, /*kitOnly=*/false, v.getOstream()),
          DiagID::ErrEmbModuleGeneric, "broken module found");

  return;
}

void llvm::verifyBitCodeAttr(KitVerifier &v, const GlobalVariable &g,
                             const TTID &tt) {
  GVAttrKind attr = GVAttrKind::BitCode;

  for (GVAttrKind a : {GVAttrKind::DeviceCode, GVAttrKind::KernelProperties})
    v.check(!hasAttr(g, a), g, DiagID::ErrAttrNotCompatible, attr, a);

  v.check(generatesEmbBC(tt), g, DiagID::ErrAttrBadValue, attr,
          DiagMessage::errTTEmbBC);

  // Check the global variable to which the attribute is attached.
  v.check(g.hasName(), g, DiagID::ErrAttrGlobalNoName, attr);
  v.check(isByteArrayTy(g.getValueType()), g, DiagID::ErrAttrGlobalBadType,
          attr);
  if (!v.check(g.hasInitializer(), g, DiagID::ErrAttrGlobalNoInit, attr))
    return;

  const Constant *init = g.getInitializer();
  if (!v.check(isa<ConstantDataArray>(init), g, DiagID::ErrAttrGlobalBadInit,
               attr, "Must be a constant data array"))
    return;

  StringRef bc = cast<ConstantDataArray>(init)->getAsString();
  if (!v.check(isBitcode(bc.bytes_begin(), bc.bytes_end()), g,
               DiagID::ErrAttrGlobalBadInit, attr, "Does not contain bitcode"))
    return;

  LLVMContext &ctx = g.getContext();
  std::unique_ptr<MemoryBuffer> buf = MemoryBuffer::getMemBuffer(bc);
  Expected<std::unique_ptr<Module>> mOrErr = parseBitcodeFile(*buf, ctx);
  if (!mOrErr) {
    handleAllErrors(mOrErr.takeError(), [&](ErrorInfoBase &e) {
      v.check(false, g, DiagID::ErrAttrGlobalBadInit, attr,
              "Could not parse bitcode");
    });
    return;
  }

  verifyEmbModule(v, *mOrErr.get(), tt);
}

void llvm::verifyDeviceCodeAttr(KitVerifier &v, const GlobalVariable &g,
                                const TTID &tt) {
  GVAttrKind attr = GVAttrKind::DeviceCode;

  for (GVAttrKind a : {GVAttrKind::BitCode, GVAttrKind::KernelProperties})
    v.check(!hasAttr(g, a), g, DiagID::ErrAttrNotCompatible, attr, a);

  v.check(generatesEmbBC(tt), g, DiagID::ErrAttrBadValue, attr,
          DiagMessage::errTTEmbBC);

  // Check the global variable to which the attribute is attached.
  v.check(g.hasName(), g, DiagID::ErrAttrGlobalNoName, attr);
  v.check(isByteArrayTy(g.getValueType()), g, DiagID::ErrAttrGlobalBadType,
          attr);
  if (!v.check(g.hasInitializer(), g, DiagID::ErrAttrGlobalNoInit, attr))
    return;

  const Constant *init = g.getInitializer();
  v.check(isa<ConstantDataArray>(init) || init->isZeroValue(), g,
          DiagID::ErrAttrGlobalBadInit, attr,
          "Must be a constant data array or zero-initialized");
}

void llvm::verifyKernelPropertiesAttr(KitVerifier &v, const GlobalVariable &g,
                                      const TTID &tt, const StringRef &name) {
  GVAttrKind attr = GVAttrKind::KernelProperties;

  for (GVAttrKind a : {GVAttrKind::BitCode, GVAttrKind::DeviceCode})
    v.check(!hasAttr(g, a), DiagID::ErrAttrNotCompatible, attr, a);

  v.check(generatesEmbBC(tt), g, DiagID::ErrAttrBadValueAt, attr, 0,
          DiagMessage::errTTEmbBC);
  v.check(name.size(), g, DiagID::ErrAttrBadValueAt, attr, 1,
          "Kernel name cannot be empty");

  // Check the global variable to which the attribute has been attached.
  if (!v.check(g.hasInitializer(), g, DiagID::ErrAttrGlobalNoInit, attr))
    return;

  const Constant *init = g.getInitializer();
  if (!v.check(isa<ConstantStruct>(init) || init->isZeroValue(), g,
               DiagID::ErrAttrGlobalBadInit, attr,
               "Must be a constant struct or zero-initialized"))
    return;

  const Module *hostM = g.getParent();
  Expected<std::unique_ptr<Module>> embMOrErr = getEmbModule(tt, *hostM);
  if (!embMOrErr) {
    handleAllErrors(embMOrErr.takeError(), [&](ErrorInfoBase &e) {
      v.check(false, g, DiagID::ErrAttrGeneric, attr,
              "requires valid embedded bitcode");
    });
    return;
  }

  // Since the embedded bitcode global variable is removed after the embedded
  // device code is generated, the module may not contain an embedded module
  // at all.
  if (std::unique_ptr<Module> embM = std::move(embMOrErr.get()))
    v.check(embM->getFunction(name), g, DiagID::ErrAttrBadValueAt, attr, 1,
            "Kernel function does not exist in embedded module");
}
