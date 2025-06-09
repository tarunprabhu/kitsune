//=-- kitmenc.cpp - Kitsune tool to encode a module as embedded bitcode -----=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tool embeds an LLVM module into an empty "host" module as Kitsune
// would embed bitcode. The host module is rendered to stdout as LLVM assembly.
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/Tapir/CommandLine.h"
#include "llvm/Frontend/Tapir/Tapir.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/KitsuneUtils.h"

using namespace llvm;
using namespace llvm::sys;

cl::OptionCategory KitMEnc("kitmenc Options");

static cl::opt<std::optional<TapirTargetID>, false, cl::TapirTargetIDParser>
    clTapir(
        "generator",
        cl::desc(
            "The tapir target that is to have generated the embedded bitcode"),
        cl::value_desc("<tapir target>"), cl::init(TapirTargetID::Cuda),
        cl::cat(KitMEnc));

static cl::opt<std::string> clInFile(cl::Positional,
                                     cl::desc("<input bitcode file>"),
                                     cl::init("-"), cl::value_desc("filename"),
                                     cl::cat(KitMEnc));

int main(int argc, char *argv[]) {
  InitLLVM X(argc, argv);
  cl::HideUnrelatedOptions(KitMEnc);
  cl::ParseCommandLineOptions(
      argc, argv,
      "Embed an LLVM module into a empty \"host\" module. Render the \"host\" "
      "module to stdout as LLVM assembly");

  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  LLVMContext ctx;
  SMDiagnostic err;
  std::unique_ptr<Module> embM = parseIRFile(clInFile, err, ctx);
  if (!embM) {
    err.print(argv[0], errs());
    return 1;
  }

  Module hostM("", ctx);
  (void)createEmbeddedBC(*embM, *clTapir, hostM);

  outs() << hostM << "\n";

  return 0;
}
