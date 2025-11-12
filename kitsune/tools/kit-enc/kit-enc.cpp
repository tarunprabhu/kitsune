//===- kit-enc.cpp - Kitsune tool to encode a module as embedded bitcode --===//
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

#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/TTOptions.h"
#include "kitsune/Core/Tapir.h"
#include "kitsune/Support/CommandLineUtils.h"
#include "kitsune/Support/TTUtils.h"
#include "kitsune/Support/ToString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::OptionCategory catKitEnc("kit-enc Options");

static cl::opt<std::string>
    clModuleName("name", cl::init(""), cl::value_desc("name"),
                 cl::desc("Override the encoded module name"));

static cl::opt<bool>
    clSkipMetadata("skip-metadata", cl::init(false),
                   cl::desc("Do not add metadata to the device module"));

static cl::opt<std::string> clInFile(cl::Positional,
                                     cl::desc("<input bitcode file>"),
                                     cl::init("-"), cl::value_desc("filename"),
                                     cl::cat(catKitEnc));

// The default tapir target to set on the encoded global variable if one has
// not been specified.
static constexpr TTID ttDefault = TTID::Cuda;

static void setupCommandLineOptions() {
  // We need this because the constructor of the TTOptions object reads
  // the -fp-contract option which requires registering the codegen flags.
  static codegen::RegisterCodeGenFlags cgf;

  cl::HideUnrelatedOptions(catKitEnc);

  StringRef optTapir = "tapir";
  clSetOptionVisible(optTapir);
  clSetOptionDescription(optTapir,
                         "The tapir target to attach to the embedded bitcode");
}

int main(int argc, char *argv[]) {
  InitLLVM X(argc, argv);

  setupCommandLineOptions();
  cl::ParseCommandLineOptions(
      argc, argv,
      "Embed an LLVM module into an empty \"host\" module. Render the \"host\" "
      "module to stdout as LLVM assembly");

  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  std::optional<TTOptions> tto =
      TTOptions::createFromCommandLine(OptznLevel::O0);
  TTID tt = tto ? tto->getTTID() : ttDefault;
  if (not doesTTGenEmbBC(tt)) {
    WithColor::error() << "'" << tt
                       << "' tapir target does not generate embedded bitcode\n";
    return 2;
  }

  LLVMContext ctx;
  SMDiagnostic err;
  std::unique_ptr<Module> embM = parseIRFile(clInFile, err, ctx);
  if (!embM) {
    err.print(argv[0], errs());
    return 1;
  }

  if (clModuleName.getNumOccurrences())
    embM->setModuleIdentifier(clModuleName);

  if (not clSkipMetadata)
    (void)addDeviceModuleMetadata(tt, *embM);

  Module hostM("", ctx);
  (void)createEmbBCGlobal(*embM, tt, hostM);
  (void)createEmbFBGlobal(tt, hostM);

  outs() << hostM;

  return 0;
}
