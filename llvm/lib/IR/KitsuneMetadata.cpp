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
static constexpr const char *mdKitsuneBC = "kitsune.bc";

/// Metadata that indicates that a global variable contains the fat binary to
/// be used by one of Kitsune's GPU runtimes. This contains a single integer
/// value which is the id of the tapir target that created this metadata.
static constexpr const char *mdKitsuneFB = "kitsune.fb";

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

std::optional<TapirTargetID> llvm::getKitsuneTTMD(const GlobalVariable &g) {
  if (MDNode *md = g.getMetadata(mdKitsuneBC))
    return ::getKitsuneTTMD(*md);
  else if (MDNode *md = g.getMetadata(mdKitsuneFB))
    return ::getKitsuneTTMD(*md);
  return std::nullopt;
}
