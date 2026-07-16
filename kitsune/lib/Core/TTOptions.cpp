//===- TTOptions.cpp - Options for the tapir targets ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation for the TTOptions object. Also contains any command line
// options shared by some or all tapir targets.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/TTOptions.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/KitOptions.h"
#include "kitsune/Core/TTUtils.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "kitsune/Support/OptznLevelUtils.h"
#include "kitsune/Support/OstreamUtils.h"
#include "kitsune/Support/ToString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;

// The options here are those that are used to initialize the TTOptions object
// *only*. Several tapir targets have additional command line options that can
// be used to tweak their behavior. Those are intended for experimentation. If
// any are deemed to be generally useful, they should be added here and a
// corresponding frontend option should be created for them.

// -------------------- options common to all tapir targets --------------------

static cl::opt<TTID>
    clTapir("tapir", cl::desc("The primary tapir target"), cl::init(TTID::Nolo),
            cl::value_desc("target"), cl::cat(cl::catKitClOpts),
            cl::values(clEnumValN(TTID::Nolo, "nolo", ""),
                       clEnumValN(TTID::Serial, "serial", ""),
                       clEnumValN(TTID::Cuda, "cuda", ""),
                       clEnumValN(TTID::Custom, "custom", ""),
                       clEnumValN(TTID::Hip, "hip", ""),
                       clEnumValN(TTID::Lambda, "lambda", ""),
                       clEnumValN(TTID::OMPTask, "omptask", ""),
                       clEnumValN(TTID::OpenCilk, "opencilk", ""),
                       clEnumValN(TTID::OpenMP, "openmp", ""),
                       clEnumValN(TTID::Pthreads, "pthreads", ""),
                       clEnumValN(TTID::Qthreads, "qthreads", ""),
                       clEnumValN(TTID::Realm, "realm", "")));

/// This was the option originally in tapir, but in Kitsune, we prefer to use
/// --tapir instead.
static cl::alias clTapirTarget("tapir-target", cl::desc("Alias for --tapir"),
                               cl::aliasopt(clTapir),
                               cl::cat(cl::catKitClOpts));

static cl::opt<std::string> clLLD("tapir-lld", cl::init(""),
                                  cl::desc("Path to LLD"),
                                  cl::cat(cl::catKitClOpts));

// ------------------ options common to the GPU tapir targets ------------------

static cl::opt<bool>
    clGPUPrefetch("tapir-gpu-prefetch",
                  cl::init(KitOptions::defaultGPUPrefetch),
                  cl::desc("Enable generation of calls to prefetch managed "
                           "memory between host and device"),
                  cl::cat(cl::catKitClOpts));

// ------------------------- cuda tapir target options -------------------------

static const std::string clCudaArchHelp = join_items(
    KitOptions::defaultCudaArch, "NVIDIA GPU architecture (default = ", ")");
static cl::opt<std::string>
    clCudaArch("tapir-cuda-arch", cl::init(KitOptions::defaultCudaArch.str()),
               cl::desc(clCudaArchHelp), cl::cat(cl::catKitClOpts));

static cl::opt<std::string>
    clCudaVirtArch("tapir-cuda-virt-arch", cl::init(""),
                   cl::desc("NVIDIA GPU virtual architecture"),
                   cl::cat(cl::catKitClOpts));

static cl::opt<std::string> clCudaFeatures(
    "tapir-cuda-features", cl::init(""),
    cl::desc("The target features to use in the cuda tapir target"),
    cl::cat(cl::catKitClOpts));

static cl::opt<std::string>
    clCudaRuntimeBCFile("tapir-cuda-runtime-bc", cl::init(""),
                        cl::desc("Path to the cuda runtime bitcode file"),
                        cl::cat(cl::catKitClOpts));

// ----------------------- 'custom' tapir target options -----------------------

static cl::opt<std::string>
    clCustomTTPlugin("tapir-plugin", cl::init(""),
                     cl::desc("Load a plugin containing a custom tapir target "
                              "from a shared object file"),
                     cl::cat(cl::catKitClOpts));

// ------------------------- hip tapir target options -------------------------

static const std::string clHipArchHelp = join_items(
    KitOptions::defaultHipArch, "AMD GPU architecture (default = ", ")");
static cl::opt<std::string>
    clHipArch("tapir-hip-arch", cl::init(KitOptions::defaultHipArch.str()),
              cl::desc(clHipArchHelp), cl::cat(cl::catKitClOpts));

static const std::string clHipSRAMECCHelp = join_items(
    toString(KitOptions::defaultHipSRAMECC),
    "Whether to enable the sramecc target feature (default = '", "')");
static cl::opt<MaybeBool> clHipSRAMECC(
    "tapir-hip-sramecc", cl::init(KitOptions::defaultHipSRAMECC),
    cl::desc(clHipSRAMECCHelp),
    cl::values(
        clEnumValN(MaybeBool::Off, "off", "Set the sramecc- target feature"),
        clEnumValN(MaybeBool::On, "on", "Set the sramecc+ target feature"),
        clEnumValN(MaybeBool::Any, "any", "Leave the sramecc feature unset")),
    cl::cat(cl::catKitClOpts));

static const std::string clHipXnackHelp =
    join_items(toString(KitOptions::defaultHipXnack),
               "Whether to enable the xnack target feature (default = '", "')");
static cl::opt<MaybeBool> clHipXnack(
    "tapir-hip-xnack", cl::init(KitOptions::defaultHipXnack),
    cl::desc(clHipXnackHelp),
    cl::values(
        clEnumValN(MaybeBool::Off, "off", "Set the xnack- target feature"),
        clEnumValN(MaybeBool::On, "on", "Set the xnack+ target feature"),
        clEnumValN(MaybeBool::Any, "any", "Leave the xnack feature unset")),
    cl::cat(cl::catKitClOpts));

static cl::opt<std::string> clHipFeatures(
    "tapir-hip-features", cl::init(""),
    cl::desc("The target features to use in the hip tapir target"),
    cl::cat(cl::catKitClOpts));

static cl::list<std::string> clHipRuntimeBCFiles(
    "tapir-hip-runtime-bcs",
    cl::desc("The bitcode files to use in the hip tapir target"),
    cl::cat(cl::catKitClOpts), cl::CommaSeparated);

// ----------------------- opencilk tapir target options -----------------------

static cl::opt<std::string> clOpenCilkRuntimeBCFile(
    "tapir-opencilk-runtime-bc", cl::init(""),
    cl::desc("Path to the bitcode file for the OpenCilk runtime bitcode file"));

static cl::alias
    clOpenCilkRuntimeBCPath("opencilk-runtime-bc-path", cl::NotHidden,
                            cl::aliasopt(clOpenCilkRuntimeBCFile),
                            cl::desc("Alias for --tapir-opencilk-runtime-bc"));

// -----------------------------------------------------------------------------

#define CHECK(expr)                                                            \
  if (Error e = (expr))                                                        \
    return e;                                                                  \
  else
#define ELSE_CHECK(expr)                                                       \
  if (Error e = (expr))                                                        \
    return e;                                                                  \
  else
#define ELSE_SUCCESS return Error::success

// If a string option is provided exactly once, it must be non-empty
static Error validateStringOption(const cl::opt<std::string> &clOpt) {
  if (clOpt.getNumOccurrences() == 1)
    if (clOpt.empty())
      return createDiagError(DiagID::ErrOptValueInvalid, clOpt.ArgStr, clOpt);
  return Error::success();
}

// If a list option is provided exactly once, it must contain at least one
// non-empty string
static Error validateListOption(const cl::list<std::string> &clOpt) {
  auto isEmpty = [](const std::string &s) -> bool { return s.empty(); };

  if (clOpt.empty() || std::all_of(clOpt.begin(), clOpt.end(), isEmpty))
    return createDiagError(DiagID::ErrOptValueMissing, clOpt.ArgStr);
  return Error::success();
}

// The given option must be provided exactly once.
static Error validateRequiredOption(const cl::Option &clOpt) {
  if (clOpt.getNumOccurrences() != 1)
    return createDiagError(DiagID::ErrOptOccurrences, clOpt.ArgStr);
  return Error::success();
}

// The given string option must be provided exactly once. The value must be
// non-empty
static Error validateRequiredStringOption(const cl::opt<std::string> &clOpt) {
  CHECK(validateRequiredOption(clOpt))
  ELSE_CHECK(validateStringOption(clOpt))
  ELSE_SUCCESS();
}

// The given list option must be provided exactly once. The list must contain
// at least one non-empty string
static Error validateRequiredListOption(const cl::list<std::string> &clOpt) {
  if (clOpt.getNumOccurrences() == 0)
    return createDiagError(DiagID::ErrOptOccurrences, clOpt.ArgStr);
  else if (Error e = validateListOption(clOpt))
    return e;
  else
    return Error::success();
}

static Error validateSupportBCFiles(TTID tt, const TTOptions &tto) {
  LLVMContext ctx;
  return getSupportModule(tt, tto, ctx).takeError();
}

template <typename Container>
static SmallVector<std::string, 4> makeSmallVector(const Container &c) {
  return SmallVector<std::string, 4>(c.begin(), c.end());
}

TTOptions::TTOptions(TTID tt) : tt(tt) {}

Error TTOptions::validateCudaOptions() const {
  CHECK(validateRequiredStringOption(clCudaArch))
  ELSE_CHECK(validateRequiredStringOption(clCudaRuntimeBCFile))
  ELSE_CHECK(validateStringOption(clCudaVirtArch))
  ELSE_CHECK(validateStringOption(clCudaFeatures))
  ELSE_CHECK(validateSupportBCFiles(TTID::Cuda, *this))
  ELSE_SUCCESS();
}

Error TTOptions::validateCustomOptions() const {
  CHECK(validateRequiredStringOption(clCustomTTPlugin))
  ELSE_CHECK(TTPlugin::load(clCustomTTPlugin).takeError())
  ELSE_SUCCESS();
}

Error TTOptions::validateHipOptions() const {
  CHECK(validateRequiredStringOption(clHipArch))
  ELSE_CHECK(validateRequiredListOption(clHipRuntimeBCFiles))
  ELSE_CHECK(validateStringOption(clHipFeatures))
  ELSE_CHECK(validateSupportBCFiles(TTID::Hip, *this))
  ELSE_SUCCESS();
}

Error TTOptions::validateOpenCilkOptions() const {
  CHECK(validateRequiredStringOption(clOpenCilkRuntimeBCFile))
  ELSE_CHECK(validateSupportBCFiles(TTID::OpenCilk, *this))
  ELSE_SUCCESS();
}

Error TTOptions::validate() const {
  switch (tt) {
  case TTID::Cuda:
    return validateCudaOptions();
  case TTID::Custom:
    return validateCustomOptions();
  case TTID::Hip:
    return validateHipOptions();
  case TTID::OpenCilk:
    return validateOpenCilkOptions();
  case TTID::Nolo:
  case TTID::OpenMP:
  case TTID::Pthreads:
  case TTID::Qthreads:
  case TTID::Serial:
    // There are no options specific to these tapir targets that need to be
    // checked.
    return Error::success();
  case TTID::Lambda:
  case TTID::OMPTask:
  case TTID::Realm:
    // These options are not fully supported.
    break;
  }
  llvm_unreachable("TTID not handled!");
}

void TTOptions::setOptznLevelFrom(OptimizationLevel optLevel) {
  unsigned speedupLevel = optLevel.getSpeedupLevel();
  unsigned sizeLevel = optLevel.getSizeLevel();
  OptznLevel optznLevel = createOptznLevelFrom(speedupLevel, sizeLevel);

  setOptznLevel(optznLevel);
}

std::optional<TTOptions>
TTOptions::createFromSharedCommandLineOptions(OptznLevel optznLevel) {
  if (clTapir.getNumOccurrences())
    return TTOptions(clTapir);
  return std::nullopt;
}

std::optional<TTOptions>
TTOptions::createFromCommandLine(OptznLevel optznLevel) {
  if (!clTapir.getNumOccurrences())
    return std::nullopt;

  TTOptions tto(clTapir);

  // Set common tapir target options
  tto.optLevel = optznLevel;
  tto.fpOpFusionMode = codegen::getFuseFPOps();
  tto.lld = clLLD;
  tto.gpuPrefetch = clGPUPrefetch;

  // Set cuda tapir target options
  tto.cudaArch = clCudaArch;
  tto.cudaVirtArch = clCudaVirtArch;
  tto.cudaTargetFeatures = clCudaFeatures;
  tto.cudaRuntimeBCFile = clCudaRuntimeBCFile;

  // Set 'custom' tapir target options
  if (clCustomTTPlugin.getNumOccurrences()) {
    // If the plugin could not be loaded, ignore the error here. In this case,
    // the validate() method will be called by users before using this object,
    // at which time the error will be caught and returned.
    if (Expected<TTPlugin> ttPlugin = TTPlugin::load(clCustomTTPlugin))
      tto.ttPlugin = *ttPlugin;
    else
      (void)toString(ttPlugin.takeError());
  }

  // Set hip tapir target options
  tto.hipArch = clHipArch;
  tto.hipSRAMECC = clHipSRAMECC;
  tto.hipXnack = clHipXnack;
  tto.hipTargetFeatures = clHipFeatures;
  tto.hipRuntimeBCFiles = makeSmallVector(clHipRuntimeBCFiles);

  // Set opencilk tapir target options
  tto.openCilkRuntimeBCFile = clOpenCilkRuntimeBCFile;

  return tto;
}

std::optional<TTOptions>
TTOptions::createFromCommandLine(unsigned speedupLevel) {
  return createFromCommandLine(createOptznLevelFrom(speedupLevel));
}

std::optional<TTOptions> TTOptions::createFromCommandLine(char optLevel) {
  return createFromCommandLine(createOptznLevelFrom(optLevel));
}

std::optional<TTOptions> TTOptions::create(const KitOptions &kitOpts,
                                           OptznLevel optznLevel,
                                           FPOpFusionMode fpOpFusionMode) {
  if (!kitOpts.getTTID())
    return std::nullopt;

  TTOptions tto(*kitOpts.getTTID());

  // Set common tapir target options.
  tto.fpOpFusionMode = fpOpFusionMode;
  tto.optLevel = optznLevel;
  tto.lld = kitOpts.getLLD();

  // Set tapir target options shared by GPU-centric tapir targets.
  tto.gpuPrefetch = kitOpts.getGPUPrefetch();

  // Set cuda tapir target options.
  tto.cudaArch = kitOpts.getCudaArch();
  tto.cudaVirtArch = kitOpts.getCudaVirtArch();
  tto.cudaTargetFeatures = kitOpts.getCudaFeatures();
  tto.cudaRuntimeBCFile = kitOpts.getCudaRuntimeBCFile();

  // Set 'custom' tapir target options.
  if (kitOpts.getTTPlugin().size()) {
    if (Expected<TTPlugin> ttPlugin = TTPlugin::load(kitOpts.getTTPlugin()))
      tto.ttPlugin = *ttPlugin;
    else
      llvm_unreachable("Tapir target plugin load failure not caught earlier");
  }

  // Set hip tapir target options.
  tto.hipArch = kitOpts.getHipArch();
  tto.hipSRAMECC = kitOpts.getHipSRAMECC();
  tto.hipXnack = kitOpts.getHipXnack();
  tto.hipTargetFeatures = kitOpts.getHipFeatures();
  tto.hipRuntimeBCFiles = makeSmallVector(kitOpts.getHipRuntimeBCFiles());

  // Set opencilk tapir target options.
  tto.openCilkRuntimeBCFile = kitOpts.getOpenCilkRuntimeBCFile();

  return tto;
}

void TTOptions::print(raw_ostream &os, bool all) const {
  os << "Tapir target options:\n";
  os << "  Primary:                 " << tt << "\n";
  os << "  Optimization level:      " << getOptznLevel() << "\n";
  os << "  FP fusion:               " << getFPOpFusionMode() << "\n";
  if (all || tt == TTID::Cuda || tt == TTID::Hip) {
    os << "  GPU prefetch:            " << getGPUPrefetch() << "\n";
  }
  if (all || tt == TTID::Cuda) {
    os << "  Cuda arch:               " << getCudaArch() << "\n";
    os << "  Cuda virtual arch:       " << getCudaVirtArch() << "\n";
    os << "  Cuda target features:    " << getCudaTargetFeatures() << "\n";
    os << "  Cuda bitcode file:       " << getCudaRuntimeBCFile() << "\n";
  }
  if (all || tt == TTID::Custom) {
    if (std::optional<TTPlugin> plugin = getTTPlugin()) {
      StringRef name = plugin->getName();
      StringRef version = plugin->getVersion();
      os << "  Custom plugin:           " << name << " " << version << "\n";
      os << "  Custom plugin file:      " << plugin->getFile() << "\n";
    }
  }
  if (all || tt == TTID::Hip) {
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
  if (all || tt == TTID::OpenCilk) {
    os << "  Opencilk bitcode file:   " << getOpenCilkRuntimeBCFile() << "\n";
  }
}
