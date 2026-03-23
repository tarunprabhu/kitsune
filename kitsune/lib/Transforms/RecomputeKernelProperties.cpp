//===- RecomputeKernelProperties.cpp - Recompute kernel properties --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Some metadata about the kernel being launched is passed to the launch calls.
// These may be used to determine launch parameters. Currently, this metadata
// includes information about the instruction mix within the kernel - the
// numbers of floating point, integer and memory operations in the kernel's IR.
// In the future, this could be expanded to include anything else that could be
// useful.
//
// This pass is run as late in the pipeline as possible to allow for all the
// optimizations to be run on the embedded bitcode. The pass updates the
// initializers of the global variables containing the properties of the
// kernels.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/RecomputeKernelProperties.h"
#include "kitsune/Analysis/TTObjectsAnalysis.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/GVAttrs.h"
#include "kitsune/Core/KernelProperties.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"

#define DEBUG_TYPE "kit-kernel-properties"

using namespace llvm;

PreservedAnalyses
RecomputeKernelPropertiesPass::run(Module &m, ModuleAnalysisManager &mam) {
  // If no primary tapir target has been set, the tapir target options will
  // not have been set, so there is nothing that we can do.
  const TTObjects &ttObjs = mam.getResult<TTObjectsAnalysis>(m);
  if (not ttObjs.hasTTID())
    return PreservedAnalyses::all();

  Expected<EmbModulesMapTy> embMsOrErr = getEmbModules(m);
  if (not embMsOrErr)
    exitOnError(embMsOrErr.takeError());

  EmbModulesMapTy embMs = std::move(*embMsOrErr);
  for (GlobalVariable &g : m.globals()) {
    if (hasKernelPropertiesAttr(g)) {
      std::optional<TTID> tt = getTTIDFromKernelPropertiesAttr(g);
      assert(tt && "Expected TTID name from kernel properties attribute");

      std::optional<StringRef> kname = getNameFromKernelPropertiesAttr(g);
      assert(kname && "Expected kernel name from kernel properties attribute");

      assert(embMs.find(*tt) != embMs.end() &&
             "Embedded module for tapir target not found");
      Function *kf = embMs.at(*tt)->getFunction(*kname);
      assert(kf && "Could not find kernel function being launched");

      ConstantStruct *c = getKernelPropertiesConstant(*kf);
      g.setInitializer(c);

      LLVM_DEBUG(dbgs() << "\tproperties:\n"
                        << "\t  " << kname << "\n"
                        << "\t    memory ops: " << c->getOperand(0) << "\n"
                        << "\t    fp ops:     " << c->getOperand(1) << "\n"
                        << "\t    int ops:    " << c->getOperand(2) << "\n"
                        << "\t    other ops:  " << c->getOperand(3) << "\n");
    }
  }

  // At best, the initializers of one or more globals will have changed, but
  // nothing else, so all analyses on the module remain valid.
  return PreservedAnalyses::all();
}
