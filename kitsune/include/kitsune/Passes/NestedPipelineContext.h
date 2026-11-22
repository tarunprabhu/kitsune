//===- NestedPipelineContext.h - Nested pass pipelines ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Class that wraps a number of analysis managers and a pass builder. This is
// intended for use when a pass pipeline is constructed within another pass,
// such as for embedded bitcode passes.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_PASSES_NESTED_PIPELINE_CONTEXT_H
#define KITSUNE_PASSES_NESTED_PIPELINE_CONTEXT_H

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/IOSandbox.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class NestedPipelineContext {
private:
  // We need to disable the sandbox before creating a pass builder. We need to
  // figure out why.
  sys::sandbox::ScopedSetting ioSandboxScope;

  // The analysis managers must be declared in this order so that they are
  // destroyed in the correct order due to inter-analysis-manager references
  LoopAnalysisManager lam;
  FunctionAnalysisManager fam;
  CGSCCAnalysisManager cgam;
  ModuleAnalysisManager mam;
  PassInstrumentationCallbacks pic;

  PassBuilder pb;

public:
  NestedPipelineContext(TargetMachine *tm = nullptr,
                        PipelineTuningOptions pto = PipelineTuningOptions())
      : ioSandboxScope(sys::sandbox::scopedDisable()), pb(tm, pto) {
    // It is not clear why we need to disable the sandbox before creating a pass
    // builder. Should really figure it out.
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    if (tm)
      tm->registerPassBuilderCallbacks(pb);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
  }

  LoopAnalysisManager &getLoopAnalysisManager() { return lam; }
  FunctionAnalysisManager &getFunctionAnalysisManager() { return fam; }
  CGSCCAnalysisManager &getCGSCCAnalysisManager() { return cgam; }
  ModuleAnalysisManager &getModuleAnalysisManager() { return mam; }
  PassInstrumentationCallbacks &getPassInstrumentationCallbacks() {
    return pic;
  }
  PassBuilder &getPassBuilder() { return pb; }
};

} // namespace llvm

#endif // KITSUNE_PASSES_NESTED_PIPELINE_CONTEXT_H
