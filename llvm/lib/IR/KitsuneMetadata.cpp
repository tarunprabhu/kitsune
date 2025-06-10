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
constexpr const char *mdKitsuneBC = "kitsune.bc";

/// Metadata that indicates that a global variable contains the fat binary to
/// be used by one of Kitsune's GPU runtimes. This contains a single integer
/// value which is the id of the tapir target that created this metadata.
constexpr const char *mdKitsuneFB = "kitsune.fb";

/// Metadata that indicates that a global variable contains metadata about a
/// kernel function. This metadata currently includes counts for various
/// instruction kinds in the function, but could be expanded to include other
/// data that could be useful for the runtime.
constexpr const char *mdKitsuneKernelMD = "kitsune.kernel.md";

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
