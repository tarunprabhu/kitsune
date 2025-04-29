//===- TapirTargetOptions.cpp - Options for the tapir targets -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation for the TapirTargetOptions object. Also contains any command
// line options shared by some or all tapir targets.
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/Tapir/TapirTargetOptions.h"
#include "kitsune/Config/config.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Frontend/Driver/KitsuneOptions.h"
#include "llvm/Frontend/Tapir/CommandLine.h"
#include "llvm/Frontend/Tapir/OptLevelUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Process.h"

using namespace llvm;

namespace {

/// Parser for the -tapir-target option to create an optional TapirTargetID.
struct TapirTargetIDParser : public cl::parser<std::optional<TapirTargetID>> {
  TapirTargetIDParser(
      cl::opt<std::optional<TapirTargetID>, false, TapirTargetIDParser> &opt)
      : parser(opt) {}
  bool parse(cl::Option &opt, StringRef name, StringRef val,
             std::optional<TapirTargetID> &result) {
    result = std::nullopt;
    if (ErrorOr<TapirTargetID> tt = parseTapirTarget(val)) {
      result = *tt;
    } else {
      opt.error("invalid value '" + val + "' in '" + name + "'");
    }
    return !result.has_value();
  }
};

} // namespace

namespace llvm {

// The options here are those that are used to initialize the TapirTargetOptions
// object *only*. Most tapir targets have additional command line options that
// can be used to tweak its behavior.

// -------------------- options common to all tapir targets --------------------

static cl::opt<std::optional<TapirTargetID>, false, TapirTargetIDParser>
    clTapirTarget("tapir-target", cl::desc("Target runtime for Tapir"),
                  cl::init(std::optional<TapirTargetID>()));

static cl::alias clTapir("tapir", cl::desc("Alias for --tapir-target"),
                         cl::aliasopt(clTapirTarget));

static cl::opt<bool>
    clTapirVerbose("tapir-verbose", cl::init(false), cl::NotHidden,
                   cl::desc("Enable verbose mode in all tapir targets"));

static cl::opt<bool>
    clKitrtVerbose("kitrt-verbose", cl::init(false), cl::NotHidden,
                   cl::desc("Enable verbose mode in kitsune's runtime"));

static cl::opt<std::string> clLLD("tapir-lld", cl::init(""), cl::NotHidden,
                                  cl::desc("Path to LLD"));

// ------------------ options common to the GPU tapir targets ------------------

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

static cl::opt<bool>
    clGPUPrefetch("tapir-gpu-prefetch",
                  cl::desc("Enable generation of calls to prefetch managed "
                           "memory between host and device"),
                  cl::init(KitsuneOptions::defaultGPUPrefetch));

// ------------------------- cuda tapir target options -------------------------

static cl::opt<std::string> clCudaArch(
    "tapir-cuda-arch", cl::init(KitsuneOptions::defaultCudaArch.str()),
    cl::NotHidden,
    cl::desc(llvm::join_items("", "NVIDIA GPU architecture (default = ",
                              KitsuneOptions::defaultCudaArch, ")")));

static cl::opt<std::string>
    clCudaVirtArch("tapir-cuda-virt-arch", cl::init(""), cl::NotHidden,
                   cl::desc("NVIDIA GPU virtual architecture"));

static cl::opt<std::string> clCudaFeatures(
    "tapir-cuda-features", cl::init(""), cl::NotHidden,
    cl::desc("The target features to use in the cuda tapir target"));

static cl::opt<std::string>
    clCudaRuntimeBCFile("tapir-cuda-runtime-bc", cl::init(""), cl::NotHidden,
                        cl::desc("Path to the cuda runtime bitcode file"));

// ------------------------- hip tapir target options -------------------------

static cl::opt<std::string>
    clHipArch("tapir-hip-arch", cl::init(KitsuneOptions::defaultHipArch.str()),
              cl::NotHidden,
              cl::desc(llvm::join_items("", "AMD GPU architecture (default = ",
                                        KitsuneOptions::defaultHipArch, ")")));

// FIXME: The default here should be 'any', not on. See the documentation of the
// corresponding frontend option in clang for more details.
static cl::opt<MaybeBool> clHipSRAMECC(
    "tapir-hip-sramecc", cl::init(KitsuneOptions::defaultHipSRAMECC),
    cl::NotHidden, cl::desc("Whether to enable the sramecc target feature"),
    cl::values(
        clEnumValN(MaybeBool::Off, "off", "Set the sramecc- target feature"),
        clEnumValN(MaybeBool::On, "on", "Set the sramecc+ target feature"),
        clEnumValN(MaybeBool::Any, "any", "Leave the sramecc feature unset")));

// FIXME: The default here should be 'any', not on. See the documentation of the
// corresponding frontend option in clang for more details.
static cl::opt<MaybeBool> clHipXnack(
    "tapir-hip-xnack", cl::init(KitsuneOptions::defaultHipXnack), cl::NotHidden,
    cl::desc("Whether to enable the xnack target feature"),
    cl::values(
        clEnumValN(MaybeBool::Off, "off", "Set the xnack- target feature"),
        clEnumValN(MaybeBool::On, "on", "Set the xnack+ target feature"),
        clEnumValN(MaybeBool::Any, "any", "Leave the xnack feature unset")));

static cl::opt<std::string> clHipFeatures(
    "tapir-hip-features", cl::init(""), cl::NotHidden,
    cl::desc("The target features to use in the hip tapir target"));

static cl::list<std::string> clHipRuntimeBCFiles(
    "tapir-hip-runtime-bcs",
    cl::desc("The bitcode files to use in the hip tapir target"),
    cl::CommaSeparated);

// ----------------------- opencilk tapir target options -----------------------

static cl::opt<std::string> clOpenCilkRuntimeBCPath(
    "opencilk-runtime-bc-path", cl::init(""),
    cl::desc("Path to the bitcode file for the OpenCilk runtime bitcode file"),
    cl::NotHidden);

static cl::alias
    clOpenCilkRuntimeBCFile("tapir-opencilk-runtime-bc",
                            cl::desc("Alias for --opencilk-runtime-bc-path"),
                            cl::aliasopt(clOpenCilkRuntimeBCPath));

// -----------------------------------------------------------------------------

TapirTargetOptions::TapirTargetOptions(TapirTargetID tt) : tt(tt) {}

std::optional<TapirTargetOptions>
TapirTargetOptions::createFromCommandLineOptions(OptimizationLevel optLevel) {
  if (clTapirTarget) {
    TapirTargetOptions tto(*clTapirTarget);

    // No validation of inputs is done here. This is intentional since these
    // command line options are primarily for internal use. Obviously, tools
    // such as opt use these too, but it is probably safe to assume that anyone
    // using opt directly is sufficiently expert. If they are not, well ...

    // Set common tapir target options
    tto.tapirVerbose = clTapirVerbose;
    tto.kitrtVerbose = clTapirVerbose || clKitrtVerbose;
    tto.optLevel = optLevel;
    tto.fpOpFusionMode = codegen::getFuseFPOps();
    tto.lld = clLLD;
    if (clFixedThreadsPerBlock)
      tto.fixedThreadsPerBlock = clFixedThreadsPerBlock;
    if (clMaxThreadsPerBlock)
      tto.maxThreadsPerBlock = clMaxThreadsPerBlock;
    tto.gpuPrefetch = clGPUPrefetch;

    // Set cuda tapir target options
    tto.cudaArch = clCudaArch;
    tto.cudaVirtArch = clCudaVirtArch;
    tto.cudaTargetFeatures = clCudaFeatures;
    tto.cudaRuntimeBCFile = clCudaRuntimeBCFile;

    // Set hip tapir target options
    tto.hipArch = clHipArch;
    tto.hipSRAMECC = clHipSRAMECC;
    tto.hipXnack = clHipXnack;
    tto.hipTargetFeatures = clHipFeatures;
    tto.hipRuntimeBCFiles = clHipRuntimeBCFiles;

    // Set opencilk tapir target options
    tto.openCilkRuntimeBCFile = clOpenCilkRuntimeBCPath;

    // FIXME: This is here purely for debugging because it was in HipABI.cpp
    // originally. It really should go away.
    if (std::optional<std::string> tpb =
            sys::Process::GetEnv("KITHIP_THREADS_PER_BLOCK")) {
      if (clFixedThreadsPerBlock)
        errs() << "kitsune[hipabi]: Note that KITHIP_THREADS_PER_BLOCK is "
               << "overriding command line args.\n";
      tto.fixedThreadsPerBlock = std::stoi(tpb.value());
    }
    return tto;
  }
  return std::nullopt;
}

std::optional<TapirTargetOptions>
TapirTargetOptions::createFromCommandLineOptions(unsigned speedupLevel) {
  return createFromCommandLineOptions(mapToOptimizationLevel(speedupLevel));
}

std::optional<TapirTargetOptions>
TapirTargetOptions::create(const KitsuneOptions &opts,
                           OptimizationLevel optLevel,
                           FPOpFusionMode fpOpFusionMode) {
  if (std::optional<TapirTargetID> tt = opts.getTapirTarget()) {
    TapirTargetOptions tto(*tt);

    // Set common tapir target options
    tto.tapirVerbose = opts.getTapirVerbose();
    tto.kitrtVerbose = opts.getTapirVerbose() or opts.getKitrtVerbose();
    tto.fpOpFusionMode = fpOpFusionMode;
    tto.optLevel = optLevel;
    tto.lld = opts.getLLD();
    tto.fixedThreadsPerBlock = opts.getFixedThreadsPerBlock();
    tto.maxThreadsPerBlock = opts.getMaxThreadsPerBlock();
    tto.gpuPrefetch = opts.getGPUPrefetch();

    // Set cuda tapir target options
    tto.cudaArch = opts.getCudaArch();
    tto.cudaVirtArch = opts.getCudaVirtArch();
    tto.cudaTargetFeatures = opts.getCudaFeatures();
    tto.cudaRuntimeBCFile = opts.getCudaRuntimeBCFile();

    // Set hip tapir target options
    tto.hipArch = opts.getHipArch();
    tto.hipSRAMECC = opts.getHipSRAMECC();
    tto.hipXnack = opts.getHipXnack();
    tto.hipTargetFeatures = opts.getHipFeatures();
    tto.hipRuntimeBCFiles = opts.getHipRuntimeBCFiles();

    // Set opencilk tapir target options
    tto.openCilkRuntimeBCFile = opts.getOpenCilkRuntimeBCFile();

    // FIXME: This is here purely for debugging because it was in HipABI.cpp
    // originally. It really should go away.
    if (std::optional<std::string> tpb =
            sys::Process::GetEnv("KITHIP_THREADS_PER_BLOCK")) {
      if (clFixedThreadsPerBlock)
        errs() << "kitsune[hipabi]: Note that KITHIP_THREADS_PER_BLOCK is "
               << "overriding command line args.\n";
      tto.fixedThreadsPerBlock = std::stoi(tpb.value());
    }
    return tto;
  }
  return std::nullopt;
}

std::unique_ptr<TapirTargetOptions> TapirTargetOptions::clone() const {
  TapirTargetOptions *clone = new TapirTargetOptions;
  *clone = *this;

  return std::unique_ptr<TapirTargetOptions>(clone);
}

bool TapirTargetOptions::lower() const { return tt != TapirTargetID::None; }

void TapirTargetOptions::print(raw_ostream &os, bool all) const {
  os << "'" << tt << "' tapir target options:\n";
  os << "  Compiler verbose:          " << getTapirVerbose() << "\n";
  os << "  Runtime verbose:           " << getKitrtVerbose() << "\n";
  os << "  Optimization level:        " << getOptLevel() << "\n";
  os << "  FP fusion: " << getFPOpFusionMode() << "\n";
  if (all || tt == TapirTargetID::Cuda || tt == TapirTargetID::Hip) {
    os << "  GPU fixed threads/block: " << getFixedThreadsPerBlock() << "\n";
    os << "  GPU max threads/block:   " << getMaxThreadsPerBlock() << "\n";
    os << "  GPU prefetch:            " << getGPUPrefetch() << "\n";
  }
  if (all || tt == TapirTargetID::Cuda) {
    os << "  Cuda arch:               " << getCudaArch() << "\n";
    os << "  Cuda virtual arch:       " << getCudaVirtArch() << "\n";
    os << "  Cuda target features:    " << getCudaTargetFeatures() << "\n";
    os << "  Cuda bitcode file:       " << getCudaRuntimeBCFile() << "\n";
  }
  if (all || tt == TapirTargetID::Hip) {
    os << "  Hip arch:                " << getHipArch() << "\n";
    os << "  Hip sramecc:             " << getHipSRAMECC() << "\n";
    os << "  Hip xnack:               " << getHipXnack() << "\n";
    os << "  Hip target features:     " << getHipTargetFeatures() << "\n";
    os << "  Hip bitcode files: [\n";
    for (StringRef file : getHipRuntimeBCFiles())
      os << "    " << file << "\n";
    os << "  ]";
    os << "  LLD:                     " << getLLD() << "\n";
  }
  if (all || tt == TapirTargetID::OpenCilk) {
    os << "  Opencilk bitcode file:   " << getOpenCilkRuntimeBCFile() << "\n";
  }
}

} // namespace llvm
