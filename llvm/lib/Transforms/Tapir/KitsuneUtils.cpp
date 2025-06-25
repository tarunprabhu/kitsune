//===- KitsuneUtils.cpp - Kitsune-specific utilities ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for the Kitsune-specific tapir targets. Moving these to a Kitsune
// support library is more trouble than it is worth.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/KitsuneUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Path.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"

using namespace llvm;

static void collectGlobalValues(Constant &c, std::set<GlobalValue *> &seen);

static void collectGlobalValues(GlobalVariable &g,
                                std::set<GlobalValue *> &seen) {
  seen.insert(&g);
  if (g.hasInitializer())
    collectGlobalValues(*g.getInitializer(), seen);
}

static void collectGlobalValues(GlobalIFunc &g, std::set<GlobalValue *> &seen) {
  seen.insert(&g);
  llvm_unreachable("kitsune: GNU IFUNC not yet supported");
}

static void collectGlobalValues(GlobalAlias &g, std::set<GlobalValue *> &seen) {
  seen.insert(&g);
  llvm_unreachable("kitsune: GlobalAlias not yet supported");
}

static void collectGlobalValues(BlockAddress &blkaddr,
                                std::set<GlobalValue *> &seen) {
  if (Function *f = blkaddr.getFunction())
    collectGlobalValues(*f, seen);
  if (BasicBlock *bb = blkaddr.getBasicBlock())
    collectGlobalValues(*bb, seen);
}

static void collectGlobalValues(Constant &c, std::set<GlobalValue *> &seen) {
  if (GlobalValue *g = dyn_cast<GlobalValue>(&c))
    if (seen.find(g) != seen.end())
      return;

  if (auto *f = dyn_cast<Function>(&c))
    return collectGlobalValues(*f, seen);
  else if (auto *g = dyn_cast<GlobalVariable>(&c))
    return collectGlobalValues(*g, seen);
  else if (auto *g = dyn_cast<GlobalAlias>(&c))
    return collectGlobalValues(*g, seen);
  else if (auto *g = dyn_cast<GlobalIFunc>(&c))
    return collectGlobalValues(*g, seen);
  else if (auto *blkaddr = dyn_cast<BlockAddress>(&c))
    return collectGlobalValues(*blkaddr, seen);
  else
    for (Use &op : c.operands())
      if (auto *cop = dyn_cast<Constant>(op))
        collectGlobalValues(*cop, seen);
}

void llvm::collectGlobalValues(BasicBlock &bb, std::set<GlobalValue *> &seen) {
  for (Instruction &inst : bb)
    for (Use &op : inst.operands())
      if (auto *c = dyn_cast<Constant>(&op))
        ::collectGlobalValues(*c, seen);
}

void llvm::collectGlobalValues(Function &f, std::set<GlobalValue *> &seen) {
  seen.insert(&f);
  for (BasicBlock &bb : f)
    collectGlobalValues(bb, seen);
}

void llvm::collectGlobalValues(Loop &loop, std::set<GlobalValue *> &seen) {
  // Collect the globals used in any subloops.
  for (Loop *subLoop : loop)
    for (BasicBlock *bb : subLoop->blocks())
      collectGlobalValues(*bb, seen);

  // Collect the globals used within the loop itself.
  for (BasicBlock *bb : loop.blocks())
    collectGlobalValues(*bb, seen);
}

std::string llvm::getNameForTapirLoop(const TapirLoopInfo &tl, StringRef pfx) {
  std::string buf;
  raw_string_ostream os(buf);
  const Loop *loop = tl.getLoop();
  const Function *f = loop->getHeader()->getParent();
  const Module *m = f->getParent();

  os << pfx;
  if (m->getNamedMetadata("llvm.dbg.cu") || m->getNamedMetadata("llvm.dbg")) {
    // If we have debug info in the module use the line number to name the
    // kernel. This is only to make debugging a shade easier since it makes it
    // easier to associate the kernel function with a loop in source code.
    //
    // FIXME: This is risky. In principle, in a large project, we could have
    // multiple files with the same name in different directories. There is a
    // small possibility that a forall loop occurs on exactly the same line in
    // both of these files. Ideally, we should include the full file path which
    // is guaranteed to be unique. However, that would detract from the
    // "usefulness" of this name (mainly for debugging). For now, we'll stick
    // with this until we can make some of the support tooling more robust to
    // allow us to mangle the name to avoid collisions.
    DebugLoc loc = loop->getStartLoc();
    unsigned line = loc.getLine();
    unsigned col = loc.getCol();
    StringRef filePath = loc->getFile()->getFilename();
    StringRef fileName = sys::path::filename(filePath);
    os << fileName << "_" << line << "_" << col;
  } else {
    StringRef name = f->getName();
    std::string demangledName;
    if (nonMicrosoftDemangle(name, demangledName,
                             /*CanHaveLeadingDot=*/false,
                             /*ParseParams=*/false))
      os << demangledName;
    else
      os << name;
  }

  return buf;
}

std::string llvm::getNameForDeviceModule(const Module &hostM, StringRef pfx) {
  return join_items("", pfx, sys::path::filename(hostM.getName()));
}
