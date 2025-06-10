//===- EmbBCPassUtils.cpp - Utilities for embedded bitcode passes ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Utilities shared by embedded bitcode passes.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Kitsune/EmbBCPassUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;

std::unique_ptr<Module> llvm::parseLibDeviceBCFile(StringRef file,
                                                   LLVMContext &ctx) {
  SMDiagnostic sm;
  std::unique_ptr<Module> m = parseIRFile(file, sm, ctx);
  if (not m)
    report_fatal_error(StringRef(join_items(
        "", "Failed to parse libdevice bitcode file: ", sm.getMessage())));
  return m;
}
