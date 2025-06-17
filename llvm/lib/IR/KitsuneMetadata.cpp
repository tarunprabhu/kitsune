//=- KitsuneMetadata.cpp - Helper functions for Kitsune metadata -*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper functions to add and query Kitsune-specific metadata.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/KitsuneMetadata.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Transforms/Utils/KitsuneUtils.h"

using namespace llvm;

/// Metadata that indicates that a global variable contains serialized LLVM
/// bitcode created by a tapir target. This contains a single integer value
/// which is the id of the tapir target that created this metadata.
static constexpr StringRef mdKitsuneBC = "kitsune.bc";

/// Metadata that indicates that a global variable contains the fat binary to
/// be used by one of Kitsune's GPU runtimes. This contains a single integer
/// value which is the id of the tapir target that created this metadata.
static constexpr StringRef mdKitsuneFB = "kitsune.fb";

/// Metadata that indicates that a global variable contains metadata about a
/// kernel function. This metadata currently includes counts for various
/// instruction kinds in the function, but could be expanded to include other
/// data that could be useful for the runtime.
static constexpr StringRef mdKitsuneKernelMD = "kitsune.kernel.md";

/// Name of a named metadata node containing top-level Kitsune metadata.
static constexpr StringRef mdKitsuneAnnotations = "kitsune.module.flags";

static std::optional<TapirTargetID> getKitsuneTTMD(const MDNode &md) {
  assert(md.getNumOperands() >= 1 &&
         "Expected at least one operand in Kitsune metadata");
  assert(isa<ConstantAsMetadata>(md.getOperand(0)) &&
         "First argument of Kitsune metadata node must be constant");

  auto *cmd = cast<ConstantAsMetadata>(md.getOperand(0));
  assert(isa<ConstantInt>(cmd->getValue()) &&
         "First argument of Kitsune metadata must be constant int");

  if (auto *cint = cast<ConstantInt>(cmd->getValue()))
    return TapirTargetID(cint->getSExtValue());

  return std::nullopt;
}

static bool hasKitsuneTTMD(const MDNode &md, TapirTargetID tt) {
  if (std::optional<TapirTargetID> maybeTT = getKitsuneTTMD(md))
    return *maybeTT == tt;
  return false;
}

void llvm::setKitsuneBCMD(GlobalVariable &g, TapirTargetID tt) {
  LLVMContext &ctx = g.getContext();
  Constant *c = getConstantInt(ctx, tt);
  MDNode *md = MDNode::get(ctx, ConstantAsMetadata::get(c));
  g.setMetadata(mdKitsuneBC, md);
}

void llvm::setKitsuneFBMD(GlobalVariable &g, TapirTargetID tt) {
  LLVMContext &ctx = g.getContext();
  Constant *c = getConstantInt(ctx, tt);
  MDNode *md = MDNode::get(ctx, ConstantAsMetadata::get(c));
  g.setMetadata(mdKitsuneFB, md);
}

void llvm::setKitsuneKernelMDMD(GlobalVariable &g, StringRef kname) {
  LLVMContext &ctx = g.getContext();
  Constant *c = ConstantDataArray::getString(ctx, kname, /*AddNull=*/true);
  MDNode *md = MDNode::get(ctx, ConstantAsMetadata::get(c));
  g.setMetadata(mdKitsuneKernelMD, md);
}

bool llvm::hasKitsuneBCMD(const GlobalVariable &g) {
  return g.hasMetadata(mdKitsuneBC);
}

bool llvm::hasKitsuneBCMD(const GlobalVariable &g, TapirTargetID tt) {
  if (MDNode *md = g.getMetadata(mdKitsuneBC))
    return hasKitsuneTTMD(*md, tt);
  return false;
}

bool llvm::hasKitsuneFBMD(const GlobalVariable &g) {
  return g.hasMetadata(mdKitsuneFB);
}

bool llvm::hasKitsuneFBMD(const GlobalVariable &g, TapirTargetID tt) {
  if (MDNode *md = g.getMetadata(mdKitsuneFB))
    return hasKitsuneTTMD(*md, tt);
  return false;
}

bool llvm::hasKitsuneKernelMDMD(const GlobalVariable &g) {
  return g.getMetadata(mdKitsuneKernelMD);
}

std::optional<TapirTargetID> llvm::getKitsuneBCMD(const GlobalVariable &g) {
  if (MDNode *md = g.getMetadata(mdKitsuneBC))
    return getKitsuneTTMD(*md);
  return std::nullopt;
}

std::optional<TapirTargetID> llvm::getKitsuneFBMD(const GlobalVariable &g) {
  if (MDNode *md = g.getMetadata(mdKitsuneFB))
    return getKitsuneTTMD(*md);
  return std::nullopt;
}

std::optional<StringRef> llvm::getKitsuneKernelMDMD(const GlobalVariable &g) {
  if (MDNode *md = g.getMetadata(mdKitsuneKernelMD)) {
    assert(md->getNumOperands() >= 1 &&
           "Expected at least one operand in kernel metadata");
    assert(isa<ConstantAsMetadata>(md->getOperand(0)) &&
           "First argument of kernel metadata node must be constant");

    auto *cmd = cast<ConstantAsMetadata>(md->getOperand(0));
    assert(isa<ConstantDataArray>(cmd->getValue()) &&
           "First argument of kernel metadata must contain constant data");

    auto *cda = cast<ConstantDataArray>(cmd->getValue());
    assert(cda->isCString() &&
           "First argument of kernel metadata must contain a string literal");

    return cda->getAsCString();
  }
  return std::nullopt;
}

bool llvm::hasKitsuneModuleMD(const Module &m) {
  return m.getNamedMetadata(mdKitsuneAnnotations);
}

void llvm::addKitsuneModuleMD(TapirTargetID tt, Module &m) {
  auto addOperandAt = [](NamedMDNode& nmd, unsigned i, MDNode* md) -> void {
    if (nmd.getNumOperands() > i)
      nmd.setOperand(i, md);
    else
      nmd.addOperand(md);
  };

  LLVMContext &ctx = m.getContext();
  Type *i8 = Type::getInt8Ty(ctx);
  NamedMDNode *nmd = m.getOrInsertNamedMetadata(mdKitsuneAnnotations);

  Constant *cTT = ConstantInt::get(i8, int(tt));

  addOperandAt(*nmd, 0, MDNode::get(ctx, ConstantAsMetadata::get(cTT)));
  addOperandAt(*nmd, 1, MDNode::get(ctx, MDString::get(ctx, m.getName())));
}

std::optional<TapirTargetID> llvm::getTapirTargetFromModuleMD(const Module &m) {
  if (const NamedMDNode *nmd = m.getNamedMetadata(mdKitsuneAnnotations))
    if (const auto *md = nmd->getOperand(0))
      if (const auto *cmd = dyn_cast<ConstantAsMetadata>(md->getOperand(0)))
        if (const auto *cint = dyn_cast<ConstantInt>(cmd->getValue()))
          return TapirTargetID(cint->getSExtValue());
  return std::nullopt;
}

std::optional<StringRef> llvm::getNameFromModuleMD(const Module &m) {
  if (const NamedMDNode *nmd = m.getNamedMetadata(mdKitsuneAnnotations))
    if (const auto *md = nmd->getOperand(1))
      if (const auto *mds = dyn_cast<MDString>(md->getOperand(0)))
        return mds->getString();
  return std::nullopt;
}
