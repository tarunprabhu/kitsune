//===- GPUABIOptions.cpp - Options for the GPU tapir targets --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared implementation for the GPU ABI options base class shared by the GPU
// tapir targets. Also declares command line options that are shared by the GPU
// tapir targets
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/GPUABIOptions.h"
#include "kitsune/Config/config.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<unsigned> clFixedThreadsPerBlock(
    "tapir-threads-per-block",
    cl::desc("Use a fixed number of threads per block for all GPU kernel "
             "launches unless overridden with pragmas. If this is not provided "
             "the threads per block will be calculated by Kitsune's runtime. "
             "Can be at most 1024"),
    cl::init(0));

static cl::opt<unsigned> clMaxThreadsPerBlock(
    "tapir-max-threads-per-block",
    cl::desc(
        "The maximum number of threads per block to launch. If this is not "
        "provided, Kitsune's runtime is free to launch as many threads per "
        "block as it sees fit"),
    cl::init(0));

namespace llvm {

void GPUABIOptionsBase::readClOptions() {
  TapirTargetOptions::readClOptions();

  if (clFixedThreadsPerBlock > KITSUNE_MAX_FIXED_THREADS_PER_BLOCK)
    report_fatal_error("-tapir-threads-per-block exceeds maximum value");
  else if (clFixedThreadsPerBlock)
    setFixedThreadsPerBlock(clFixedThreadsPerBlock);

  if (clMaxThreadsPerBlock)
    setMaxThreadsPerBlock(clMaxThreadsPerBlock);
}

void GPUABIOptionsBase::setOptLevel(unsigned optLevel) {
  switch (optLevel) {
  case 0:
    return setOptLevel(OptimizationLevel::O0);
  case 1:
    return setOptLevel(OptimizationLevel::O1);
  case 2:
    return setOptLevel(OptimizationLevel::O2);
  case 3:
    return setOptLevel(OptimizationLevel::O3);
  default:
    llvm_unreachable("setOptLevel: Unknown optimization level");
  }
}

} // namespace llvm
