//===- kit-linkdc.cpp - Link embedded device code -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tool simply links the embedded device code found in the given binary
// files. This is *not* intended to be a replacement for lld. Its behavior is
// closer to that of llvm-link.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/CommandLineOptions.h"
#include "kitsune/Core/EmbDeviceCodeUtils.h"
#include "kitsune/Core/Tapir.h"
#include "kitsune/Linker/DeviceCodeLinker.h"
#include "kitsune/Object/BinaryUtils.h"
#include "kitsune/Support/Error.h"
#include "kitsune/Support/StringUtils.h"
#include "kitsune/Support/TTUtils.h"
#include "kitsune/Support/ToString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

static cl::OptionCategory catKitLinkDC("Kitsune Options (kit-linkdc)");

static cl::opt<std::string> clSaveTemps(
    "save-temps", cl::init(""), cl::value_desc("dir"),
    cl::desc("Save any temporary files in the given directory. If the "
             "directory does not exist, it will be created. If a directory (or "
             "file) with the given name already exists, it will be deleted and "
             "a new directory will be created"),
    cl::cat(catKitLinkDC));

static cl::opt<bool> clFatbin(
    "fatbinary", cl::init(false),
    cl::desc("Create a fat binary from the embedded device code. If this is "
             "not provided, a static archive will be created containing the "
             "embedded device code. In either case, the linked results will be "
             "in specific sections of the output file"),
    cl::cat(catKitLinkDC));

static cl::opt<std::string> clOutFile(
    "o", cl::init("a.o"), cl::value_desc("file"),
    cl::desc("The output file. This will always be a relocatable object. The "
             "linked device code will be in specific sections of this file. If "
             "no embedded device code was found in the input, the output file "
             "will be created anyway"),
    cl::cat(catKitLinkDC));
static cl::alias clOutFileLong("output", cl::aliasopt(clOutFile),
                               cl::desc("Alias for -o"), cl::NotHidden,
                               cl::cat(catKitLinkDC));

static cl::opt<std::string>
    clTarget("target", cl::init(sys::getDefaultTargetTriple()),
             cl::value_desc("triple"),
             cl::desc("The target triple of the host object file into which "
                      "the linked device code will be embedded"),
             cl::cat(catKitLinkDC));

static cl::list<std::string> clInFiles(cl::Positional, cl::OneOrMore,
                                       cl::desc("<files>"),
                                       cl::cat(catKitLinkDC));

int main(int argc, char *argv[]) {
  InitLLVM X(argc, argv);

  cl::HideUnrelatedOptions(catKitLinkDC);
  cl::ParseCommandLineOptions(
      argc, argv, "Link the embedded device code found in object files");

  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();

  SmallVector<std::unique_ptr<MemoryBuffer>> inFiles;
  for (StringRef f : clInFiles) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> memBuf = MemoryBuffer::getFile(
        f, /*IsText=*/false, /*RequiresNullTerminator=*/false);
    if (not memBuf)
      report_error(sjoin(f, ": ", memBuf.getError()));
    inFiles.emplace_back(std::move(memBuf.get()));

    MemoryBufferRef memRef = inFiles.back()->getMemBufferRef();
    if (not isObject(memRef) and not isArchive(memRef) and not isShared(memRef))
      report_error(sjoin(f, ": file format not supported"));
  }

  DeviceCodeLinker linker;
  for (const std::unique_ptr<MemoryBuffer> &inFile : inFiles) {
    Expected<int> added = linker.add(*inFile);
    if (not added)
      report_error(added.takeError());
  }

  DeviceCodeLinker::Mode mode;
  if (clFatbin)
    mode = DeviceCodeLinker::FatBin;
  else
    mode = DeviceCodeLinker::Static;
  Expected<OwningBinary<ObjectFile>> linked = linker.link(mode, clTarget);
  if (not linked)
    report_error(linked.takeError());

  std::error_code ec;
  raw_fd_ostream fs(clOutFile, ec);
  if (ec)
    report_error(ec);

  fs << linked->getBinary()->getMemoryBufferRef().getBuffer();
  fs.close();

  return 0;
}
