//===- DeviceCode.cpp - Utilities for Kitsune's embedded device code ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeviceCode.h"
#include "Config.h"
#include "kitsune/Config/config.h"
#include "kitsune/Core/SingletonUtils.h"
#include "kitsune/Support/TTUtils.h"
#include "kitsune/Support/ToString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

using namespace lld;
using namespace lld::elf;
using namespace llvm;

namespace {

using Path = DeviceCodeCtx::Path;

static raw_ostream &err(raw_ostream &os) {
  os.flush();
  return os;
}

template <typename Arg, typename... Args>
static raw_ostream &err(raw_ostream &os, Arg &&arg, Args &&...args) {
  os << arg;
  return err(os, args...);
}

template <typename... Args> static void err(Ctx &ctx, Args &&...args) {
  std::string buf;
  raw_string_ostream os(buf);
  err(os, args...);
  ctx.e.error(buf);
}

static Path getEmptyFilePath(StringRef tempDir) {
  Path path;
  sys::path::append(path, tempDir, "empty.dat");
  return path;
}

static size_t align(size_t pos, size_t multiple) {
  return ((pos + multiple - 1) / multiple) * multiple;
}

class DeviceCodeLinker {
protected:
  Ctx &ctx;
  StringRef tempDir;
  StringRef varName;
  StringRef sectionName;
  StringRef linkedFileName;
  StringRef finalObjFileName;

protected:
  DeviceCodeLinker(Ctx &ctx, StringRef tempDir) : ctx(ctx), tempDir(tempDir) {}

  Path getLLVMTool(StringRef tool) {
    Path path;
    StringRef dirName = sys::path::parent_path(ctx.arg.progName);
    sys::path::append(path, dirName, tool);
    return path;
  }

  Path getLinkedFilePath() {
    Path path;
    sys::path::append(path, tempDir, linkedFileName);
    return path;
  }

  Path getFinalObjFilePath() {
    Path path;
    sys::path::append(path, tempDir, finalObjFileName);
    return path;
  }

  std::string getOutputTarget() {
    auto getOutputFormat = [](bool is64) -> StringRef {
      return is64 ? "elf64" : "elf32";
    };

    auto getOutputMachine = [](unsigned emachine) -> StringRef {
      switch (emachine) {
      case ELF::EM_X86_64:
        return "x86-64";
      case ELF::EM_AARCH64:
        return "aarch64";
      default:
        break;
      }
      llvm_unreachable("getOutputMachine: ELF ID not handled");
    };

    StringRef format = getOutputFormat(ctx.arg.is64);
    StringRef machine = getOutputMachine(ctx.arg.emachine);
    return join_items("-", format, machine);
  }

  /// When embedding the fat binary into an object file, _start, _end and _size
  /// symbols are generated from the <infile> argument. The name of this symbol
  /// has the form
  ///
  ///     _binary__<PATH>_<SUFFIX>
  ///
  /// where
  ///
  ///   <PATH>   is the <infile> command line argument with any characters that
  ///            are not allowed in C identifiers replaced with '_'
  ///
  ///   <SUFFIX> One of "start", "end", "size" (with the double quotes)
  ///
  std::string getSymbol(StringRef inFile, StringRef suffix) {
    SmallVector<StringRef, 4> parts;
    StringRef sep = sys::path::get_separator();
    inFile.split(parts, sep, /*MaxSplit=*/-1, /*KeepEmpty=*/false);

    std::string buf;
    raw_string_ostream os(buf);
    os << "_binary_";
    for (StringRef part : parts) {
      os << '_';
      for (char c : part)
        os << ((std::isalnum(c) || c == '_') ? c : '_');
    }
    os << '_' << suffix;
    os.flush();

    return buf;
  }

  void embedIntoObjectFile(StringRef linkedFile, StringRef objFile) {
    Path emptyFile = getEmptyFilePath(tempDir);
    std::string symStart = getSymbol(emptyFile, "start");
    std::string symEnd = getSymbol(emptyFile, "end");
    std::string symSize = getSymbol(emptyFile, "size");

    // TODO: It would be nice if we could replace this with the appropriate
    // calls to LLVM's objcopy library. That way, we would have to make this
    // ugly external process spawn call.
    Path prog = getLLVMTool("llvm-objcopy");
    std::vector<StringRef> args = {prog};

    // Tell llvm-objcopy to treat the input file as an opaque binary blob.
    // Without this, llvm-objcopy will determine the type of the input file from
    // its contents. If the input looks like an ELF file, the symbols and
    // sections in it will be copied over into the output file as is. The result
    // will be a an ELF file for the *device*. What we actually want is a valid
    // relocatable ELF object file for the *host*.
    StringRef iTarget = "--input-target=binary";
    args.push_back(iTarget);

    // Specify the output target. Without this, objcopy will attempt to infer
    // the output target type from the input file. If the input happens to be an
    // ELF file for the device - the output will be determined to be an object
    // file for device code. What we actually want is an object file for the
    // host. The output target string will be similar to "elf64-x86-64",
    // "elf64-aarch64" etc.
    std::string oTarget = join_items("=", "--output-target", getOutputTarget());
    args.push_back(oTarget);

    // Once the binary blob has been embedded into a *host* object file, a
    // single empty section .data section will have been created. The section
    // will be empty because the input file is empty. We have to use an empty
    // input file because, despite the --input-target=binary option,
    // llvm-objcopy examines the input file and, if it is determined to be a
    // valid ELF file, the output is simply a clone of the input.
    //
    // With this option, we actually set the contents of the data section to be
    // the linked device code file that is to be embedded inside the host object
    // file.
    std::string setSectData =
        join_items("=", "--update-section", ".data", linkedFile);
    args.push_back(setSectData);

    // NVIDIA/AMD's tools require the linked device code file to be in a
    // specific named section. Rename the .data section to the expected name.
    // The .data section is writeable by default. For additional safety, mark
    // the section 'readonly'.
    StringRef sectionFlags = "alloc,load,readonly,data,contents";
    std::string setSectName = join_items(
        "", "--rename-section=", ".data=", sectionName, ",", sectionFlags);
    args.push_back(setSectName);

    // When the input file is embedded into the generated object file, three
    // symbols are created marking the start, end and size of the embedded
    // object. The names of these symbols are derived from the requied <infile>
    // command line argument. Kitsune's runtime requires the linked device code
    // file to have a specific name. Rename the "start" symbol so it gets
    // linked correctly with the runtime.
    std::string redefSym = join_items("=", "--redefine-sym", symStart, varName);
    args.push_back(redefSym);

    // Of the three symbols that were automatically generated, we only need the
    // start which contains the address of the start of the linked device code.
    // To avoid having unnecessary symbols in the final executable, remove
    // these.
    std::string stripEnd = join_items("=", "--strip-symbol", symEnd);
    std::string stripSize = join_items("=", "--strip-symbol", symSize);
    args.push_back(stripEnd);
    args.push_back(stripSize);

    // The <infile> command line argument. This is the name of the input file.
    // This is an empty file. See the comments associated with the
    // --update-section llvm-objcopy command line option for a more detailed
    // explanation.
    args.push_back(emptyFile);

    // The output file. This is the path to the final host object file into
    // which the linked file is embedded.
    args.push_back(objFile);

    if (ctx.e.verbose)
      Msg(ctx) << join(args, " ");

    // The 0 values for SecondsToWait and MemoryLimit indicate that there is no
    // limit on the command execution time, nor the amount of memory it can use.
    std::string errMsg;
    if (sys::ExecuteAndWait(args[0], args,
                            /*Env=*/std::nullopt,
                            /*Redirects=*/{},
                            /*SecondsToWait=*/0, /* 0 => unlimited */
                            /*MemoryLimit=*/0,   /* 0 => unlimited */
                            &errMsg))
      err(ctx, "could not generate embedded object file: ", errMsg);
  }

  // Link the given object files into the given output file.
  virtual void link(ArrayRef<Path> objFiles, StringRef linkedFile) = 0;

public:
  virtual ~DeviceCodeLinker() = default;

  Path run(ArrayRef<Path> objFiles) {
    Path finalObjFile = getFinalObjFilePath();
    Path linkedFile = getLinkedFilePath();

    link(objFiles, linkedFile);
    if (ctx.e.errorCount == 0)
      embedIntoObjectFile(linkedFile, finalObjFile);

    return finalObjFile;
  }
};

class CudaLinker : public DeviceCodeLinker {
protected:
  virtual void link(ArrayRef<Path> files, StringRef linkedFile) override {
    // TODO: Construct this command line properly.
    StringRef prog = KITSUNE_CUDA_FATBINARY;
    std::vector<StringRef> args = {prog};
    args.push_back("--64");
    args.push_back("--create");
    args.push_back(linkedFile);

    // FIXME: For now, just assume that everything is a SASS file and compiled
    // for sm_70. This is obviously very wrong, but it's just to see if the
    // basic stuff works.
    std::vector<std::string> images;
    for (StringRef file : files) {
      images.push_back(join_items("", "--image=profile=sm_70", ",file=", file));
      args.push_back(images.back());
    }

    if (ctx.e.verbose)
      Msg(ctx) << join(args, " ");

    // The 0 values for SecondsToWait and MemoryLimit indicate that there is
    // no limit on the command execution time, nor the amount of memory it can
    // use.
    std::string errMsg;
    if (sys::ExecuteAndWait(args[0], args,
                            /*Env=*/std::nullopt,
                            /*Redirects=*/{},
                            /*SecondsToWait=*/0, /* 0 => unlimited */
                            /*MemoryLimit=*/0,   /* 0 => unlimited */
                            &errMsg))
      err(ctx, errMsg);
  }

public:
  ~CudaLinker() = default;

  CudaLinker(Ctx &ctx, StringRef tempDir) : DeviceCodeLinker(ctx, tempDir) {
    varName = cudaFatbinName;
    sectionName = cudaFatbinSection;
    linkedFileName = "kitcuda_linked.o";
    finalObjFileName = "kitcuda_fb.o";
  }
};

class HipLinker : public DeviceCodeLinker {
protected:
  virtual void link(ArrayRef<Path> objFiles, StringRef linkedFile) override {
    Path prog = getLLVMTool("ld.lld");
    std::vector<StringRef> args = {prog};
    args.push_back("-m");
    args.push_back("elf64_amdgpu");
    args.push_back("--eh-frame-hdr");
    args.push_back("--no-undefined");
    args.push_back("-shared");
    args.push_back("-o");
    args.push_back(linkedFile);
    for (StringRef objFile : objFiles)
      args.push_back(objFile);

    if (ctx.e.verbose)
      Msg(ctx) << join(args, " ");

    // It would be good if we could just call lldMain, but there is still some
    // global state that causes issues. If that is ever fixed by upstream, we
    // should be able to just call lldMain here with this argument list which
    // will save us having to spawn a process.
    //
    // The 0 values for SecondsToWait and MemoryLimit indicate that there is
    // no limit on the command execution time, nor the amount of memory it can
    // use.
    std::string errMsg;
    if (sys::ExecuteAndWait(args[0], args,
                            /*Env=*/std::nullopt,
                            /*Redirects=*/{},
                            /*SecondsToWait=*/0, /* 0 => unlimited */
                            /*MemoryLimit=*/0,   /* 0 => unlimited */
                            &errMsg))
      err(ctx, errMsg);
  }

public:
  ~HipLinker() = default;

  HipLinker(Ctx &ctx, StringRef tempDir) : DeviceCodeLinker(ctx, tempDir) {
    varName = hipFatbinName;
    sectionName = hipFatbinSection;
    linkedFileName = "kithip_linked.so";
    finalObjFileName = "kithip_fb.o";
  }
};

} // namespace

DeviceCodeCtx::DeviceCodeCtx(Ctx &ctx) : ctx(ctx) {
  // The ctx object will not have been initialized when this is called, so it
  // should not be used for anything.
}

DeviceCodeCtx::~DeviceCodeCtx() {
  if (tempDir.size() && ctx.arg.saveTempsArgs.empty())
    sys::fs::remove_directories(tempDir);
}

bool DeviceCodeCtx::createWorkingDir() {
  if (tempDir.empty()) {
    if (std::error_code ec =
            sys::fs::createUniqueDirectory("kit-lld", tempDir)) {
      err(ctx, ec.message());
      return false;
    }
  }
  return true;
}

StringRef DeviceCodeCtx::getTempFilePath(TTID tt, StringRef ext) {
  // The files containing the device code are named 1.o, 2.o etc. This is
  // determined by the number of object files that have been recorded. First,
  // create an empty string at the end of the list of object files for the
  // tapir target. That string is populated in the call to path::append.
  tempFiles[tt].emplace_back("");
  size_t idx = tempFiles[tt].size();
  Path &objFile = tempFiles[tt].back();
  sys::path::append(objFile, tempDir, std::to_string(idx));
  sys::path::replace_extension(objFile, ext);

  return objFile.str();
}

bool DeviceCodeCtx::saveDeviceCode(TTID tt, StringRef code, StringRef ext) {
  if (not createWorkingDir())
    return false;

  std::error_code ec;
  StringRef objFile = getTempFilePath(tt, ext);
  raw_fd_ostream fs(objFile, ec);
  if (ec) {
    err(ctx, "Could not create file for device code buffer: ", ec.message());
    return false;
  }
  fs << code;
  fs.close();

  return true;
}

unsigned DeviceCodeCtx::parseSectionCuda(ArrayRef<char> buf) {
  auto error = [&](StringRef msg, size_t pos) -> int {
    err(ctx, msg, " at offset ", pos);
    return 0;
  };

  // The section can be treated as an array of non-uniform structs:
  //
  //   struct {
  //     uint64_t size;  // The size, in bytes, of the code block
  //     uint32_t isPtx; // A boolean where 1 indicates that the code is PTX,
  //                     // SASS otherwise
  //     uint32_t arch;  // The architecture. If the cuda architecture is
  //                     // "sm_86", this will be 86.
  //     byte code[];    // A block of <SIZE> bytes.
  //   };
  //
  // Each struct is aligned on an 8-byte boundary.
  //
  unsigned count = 0;
  for (size_t pos = 0; pos < buf.size();) {
    if (pos + 8 > buf.size())
      return error("invalid device code section: Expected size", pos);
    uint64_t size = *reinterpret_cast<const uint64_t *>(&buf[pos]);
    pos += 8;

    if (pos + 4 > buf.size())
      return error("invalid device code section: Expected flag", pos);
    uint32_t isPtx = *reinterpret_cast<const uint32_t *>(&buf[pos]);
    pos += 4;

    if (pos + 4 > buf.size())
      return error("invalid device code section: Expected arch", pos);
    uint32_t arch = *reinterpret_cast<const uint32_t *>(&buf[pos]);
    pos += 4;

    if (pos + size > buf.size())
      return error("invalid device code section: Unexpected end of section",
                   pos);
    StringRef code(&buf[pos], size);
    pos += size;

    StringRef ext = isPtx ? "ptx" : "s";
    if (not saveDeviceCode(TTID::Cuda, code, ext))
      return 0;

    ++count;
    pos = align(pos, 8);
  }
  return count;
}

unsigned DeviceCodeCtx::parseSectionHip(ArrayRef<char> buf) {
  // The section can be treated as an array of non-uniform structs:
  //
  //   struct {
  //     uint64_t size; // The size, in bytes, of the following code block
  //     byte code[];   // A block of <SIZE> bytes.
  //   };
  //
  // Each struct is aligned on an 8-byte boundary.
  //
  unsigned count = 0;
  for (size_t pos = 0; pos < buf.size();) {
    if (pos + 8 > buf.size()) {
      err(ctx, "invalid device code section: Expected size at offset ", pos);
      return 0;
    }
    uint64_t size = *reinterpret_cast<const uint64_t *>(&buf[pos]);
    pos += 8;

    if (pos + size > buf.size()) {
      err(ctx, "invalid device code section: Unexpected end of section");
      return 0;
    }
    StringRef code(&buf[pos], size);
    pos += size;

    if (not saveDeviceCode(TTID::Hip, code, "o"))
      return 0;

    ++count;
    pos = align(pos, 8);
  }
  return count;
}

unsigned DeviceCodeCtx::parseSection(TTID tt, ArrayRef<char> buf) {
  switch (tt) {
  case TTID::Cuda:
    return parseSectionCuda(buf);
  case TTID::Hip:
    return parseSectionHip(buf);
  default:
    llvm_unreachable("DeviceCodeCtx::parseSection: TTID not handled");
  }
}

bool DeviceCodeCtx::createEmptyFile() {
  std::error_code ec;
  raw_fd_ostream nullf(getEmptyFilePath(tempDir), ec);
  if (ec) {
    err(ctx, "Could not create empty file");
    return false;
  }
  nullf.close();

  return true;
}

SmallVector<Path, 0> DeviceCodeCtx::linkAll() {
  auto getLinker = [this](TTID tt) -> std::unique_ptr<DeviceCodeLinker> {
    Ctx &ctx = this->ctx;
    StringRef tempDir = this->tempDir;
    switch (tt) {
    case TTID::Cuda:
      return std::make_unique<CudaLinker>(ctx, tempDir);
    case TTID::Hip:
      return std::make_unique<HipLinker>(ctx, tempDir);
    default:
      llvm_unreachable("DeviceCodeCtx::linkAll: TTID not handled");
    }
  };

  SmallVector<Path, 0> linkedFiles;
  if (tempFiles.empty())
    return linkedFiles;

  if (not createEmptyFile())
    return linkedFiles;

  for (TTID tt : ttsUsingEmbBC) {
    if (tempFiles.find(tt) != tempFiles.end()) {
      std::unique_ptr<DeviceCodeLinker> linker = getLinker(tt);
      const SmallVectorImpl<Path> &objFiles = tempFiles.at(tt);
      Path finalObjFile = linker->run(objFiles);

      linkedFiles.push_back(finalObjFile);
    }
  }
  return linkedFiles;
}
