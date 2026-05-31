//===- kit-sort.cpp - Kitsune tool to sort basic blocks in functions ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tool sorts the basic blocks in functions in some "reasonable" order.
// This is usually reverse postorder since that is closest to "program order",
// but it may be some hybrid ordering.
//
// This is guaranteed to produce a reliable ordering of blocks,  which is
// particularly useful during testing since transformation passes may have
// moved basic blocks in a way that makes it hard to follow the control-flow.
//
// Since this is primarily intended to aid in testing and debugging, it only
// writes LLVM assembly to stdout. We may add options to write to file, as well
// as output LLVM bitcode if a need emerges for it.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/FuncUtils.h"
#include "kitsune/Support/CommandLineUtils.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::OptionCategory catKitSort("kit-sort options");

static cl::list<std::string>
    clFuncs("funcs", cl::desc("The functions whose blocks are to be sorted"),
            cl::CommaSeparated, cl::cat(catKitSort));

static cl::opt<std::string> clInFile(cl::Positional,
                                     cl::desc("<input bitcode file>"),
                                     cl::init("-"), cl::value_desc("filename"),
                                     cl::cat(catKitSort));

static bool complain(StringRef fname) {
  emitDiagnostic(DiagID::ErrGeneric,
                 formatv("function '{0}' not found", fname));
  return false;
}

static bool checkCLFuncs(Module &m) {
  bool ok = true;
  if (clFuncs.size())
    for (const StringRef fname : clFuncs)
      if (!m.getFunction(fname))
        ok = complain(fname);
  return ok;
}

static SmallVector<Function *, 4> collectFuncs(Module &m) {
  SmallVector<Function *, 4> funcs;
  if (clFuncs.size()) {
    for (const StringRef fname : clFuncs)
      if (Function *f = m.getFunction(fname))
        if (f->size())
          funcs.push_back(f);
  } else {
    for (Function &f : m.functions())
      if (f.size())
        funcs.push_back(&f);
  }
  return funcs;
}

int main(int argc, char *argv[]) {
  InitLLVM X(argc, argv);

  cl::HideUnrelatedOptions(catKitSort);
  cl::ParseCommandLineOptions(
      argc, argv,
      "Sort function blocks in some \"reasonable\" program order\n");

  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  LLVMContext ctx;
  SMDiagnostic err;
  std::unique_ptr<Module> m = parseIRFile(clInFile, err, ctx);
  if (!m) {
    err.print(argv[0], errs());
    return 1;
  }

  // If any functions are explicitly requested, check that they exist in the
  // module. If any don't, emit an error message and exit immediately. Yes, this
  // is pretty strict, but it is, arguably, less surprising for the user.
  if (!checkCLFuncs(*m))
    return 1;

  // At this point, everything is ok. We can now do what needs to be done.
  for (Function *f : collectFuncs(*m))
    sortBasicBlocks(*f);

  outs() << *m << "\n";
  return 0;
}
