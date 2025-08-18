//===- KitsuneLoweringUtils.cpp - Utilities for Kitsune's tapir targets ---===//
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

#include "llvm/Transforms/Tapir/KitsuneLoweringUtils.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/EmbBitcodeUtils.h"
#include "kitsune/Core/EmbDeviceCodeUtils.h"
#include "kitsune/Core/ReachableGlobals.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Path.h"

using namespace llvm;

std::string llvm::getNameForTapirLoop(const Loop &loop, StringRef pfx,
                                      unsigned suffix) {
  std::string buf;
  raw_string_ostream os(buf);
  const Function &f = *loop.getHeader()->getParent();
  const Module &m = *f.getParent();

  os << pfx;
  if (m.getNamedMetadata("llvm.dbg.cu") || m.getNamedMetadata("llvm.dbg")) {
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
    DebugLoc loc = loop.getStartLoc();
    unsigned line = loc.getLine();
    unsigned col = loc.getCol();
    StringRef filePath = loc->getFile()->getFilename();
    StringRef fileName = sys::path::filename(filePath);
    os << fileName << "_" << line << "_" << col;
  } else {
    StringRef name = f.getName();
    std::string demangledName;
    if (nonMicrosoftDemangle(name, demangledName,
                             /*CanHaveLeadingDot=*/false,
                             /*ParseParams=*/false))
      os << demangledName;
    else
      os << name;
    os << "_" << suffix;
  }

  return buf;
}

static void copyNonConstGlobals(const ReachableGlobals &globals, TTID tt,
                                Intrinsic::ID copyFn, Module &m,
                                IRBuilder<> &builder) {
  const DataLayout &dl = m.getDataLayout();
  LLVMContext &ctx = m.getContext();
  Type *i64Ty = Type::getInt64Ty(ctx);
  Type *voidTy = Type::getVoidTy(ctx);
  PointerType *ptrTy = PointerType::getUnqual(ctx);

  GlobalVariable *fb = getSingletonFBGlobal(tt, m);
  assert(fb && "Singleton fat binary must exist");

  Constant *ctt = createConstInt(tt, ctx);
  for (GlobalValue *gv : globals) {
    if (auto *g = dyn_cast<GlobalVariable>(gv)) {
      if (not g->isConstant()) {
        GlobalVariable *name = createConstString(g->getName(), m);
        Type *type = g->getValueType();
        size_t size = dl.getTypeAllocSize(type);
        Constant *bytes = ConstantInt::get(i64Ty, size);

        Value *devPtr = builder.CreateIntrinsic(
            ptrTy, Intrinsic::kit_symbol_device_ptr, {ctt, fb, name});
        if (copyFn == Intrinsic::kit_symbol_memcpy_dtoh)
          (void)builder.CreateIntrinsic(voidTy, copyFn,
                                        {ctt, g, devPtr, bytes});
        else if (copyFn == Intrinsic::kit_symbol_memcpy_htod)
          (void)builder.CreateIntrinsic(voidTy, copyFn,
                                        {ctt, devPtr, g, bytes});
        else
          llvm_unreachable("copyNonConstGlobals: Invalid intrinsic");
      }
    }
  }
}

void llvm::copyNonConstGlobalsDToH(const ReachableGlobals &globals, TTID tt,
                                   Module &m, IRBuilder<> &builder) {
  copyNonConstGlobals(globals, tt, Intrinsic::kit_symbol_memcpy_dtoh, m,
                      builder);
}

void llvm::copyNonConstGlobalsHToD(const ReachableGlobals &globals, TTID tt,
                                   Module &m, IRBuilder<> &builder) {
  copyNonConstGlobals(globals, tt, Intrinsic::kit_symbol_memcpy_htod, m,
                      builder);
}
