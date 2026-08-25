//===- EmbPrepare.cpp - Prepare embedded modules for codegen --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prepare the embedded modules for code generation. This should be run
// relatively late in the pipeline. This will carry out any
// architecture-specific transformations that are unrelated to optimizations
// and that have not already been carried out by the tapir targets that created
// this module. For instance, for AMDGPU kernels, the kernel function
// arguments must be placed in a specific address space, as must any alloca's
// in the kernel functions. In some cases, the calling conventions of such
// functions must also be changed.
//
// This is mainly intended to be carried out on functions that are *not* the
// GPU entry points, i.e. the GPU "kernel" functions that are launched from the
// host, although this is not strictly enforced. The idea here is that the
// tapir target will have taken care of adding the correct attributes to the
// kernel function, but may not have done anything for the callees.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/EmbPrepare.h"
#include "EmbPrepareImpl.h"
#include "kitsune/Support/CommandLineOptions.h"

using namespace llvm;

static cl::opt<bool> clInlineAll(
    "emb-inline-all", cl::init(false), cl::Hidden,
    cl::desc("Inline all device functions in the kernel module, unless they "
             "have the 'noinline' attribute"),
    cl::cat(cl::catKitClDevOpts));

static cl::opt<bool> clInlineAllForce(
    "emb-inline-all-force", cl::init(false), cl::Hidden,
    cl::desc("Inline all device functions in the kernel module, including "
             "those that have the 'noinline' attribute"),
    cl::cat(cl::catKitClDevOpts));

bool EmbPreparePass::run(TTID tt, Module &devM, Module &hostM,
                         ModuleAnalysisManager &hostMAM) {
  detail::EmbPrepareOptions prepOpts;
  prepOpts.inlineAll = clInlineAll;
  prepOpts.inlineAllForce = clInlineAllForce;

  switch (tt) {
  case TTID::Cuda: return detail::embPrepareCuda(devM, tto, prepOpts);
  case TTID::Hip: return detail::embPrepareHip(devM, tto, prepOpts);
  default: llvm_unreachable("EmbPreparePass::run: TTID not handled");
  }
}
