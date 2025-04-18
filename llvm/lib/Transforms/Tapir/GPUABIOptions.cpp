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
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

namespace llvm {

void GPUABIOptionsBase::readClOptions() {
}


} // namespace llvm
