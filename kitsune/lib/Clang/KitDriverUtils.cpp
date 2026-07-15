//===- KitDriverUtils.cpp - Utilities for Kitsune's command-line options --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for Kitsune-specific command line options.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Clang/KitDriverUtils.h"
#include "kitsune/Config/Config.h"
#include "kitsune/Core/KitOptions.h"
#include "kitsune/Core/TTPlugin.h"
#include "kitsune/Core/TTUtils.h"
#include "kitsune/Support/FromString.h"
#include "kitsune/Support/ToString.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/OffloadArch.h"
#include "clang/Driver/OptionUtils.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Option/ArgList.h"

#include <sstream>

using namespace clang;
using namespace clang::driver;
using namespace clang::driver::options;
using namespace llvm;
using namespace llvm::opt;

using llvm::driver::KitOptions;

bool clang::driver::isKitsuneFrontend(StringRef prog) {
  // Yes, the name of the compiler is actually the "ModeSuffix". Don't ask ...
  std::string suffix =
      ToolChain::getTargetAndModeFromProgramName(prog).ModeSuffix;
  return suffix == llvm::kitCFrontend() || suffix == llvm::kitCXXFrontend() ||
         suffix == llvm::kitFortranFrontend();
}

static std::optional<std::string> getUniqueArgValue(const ArgList &args,
                                                    OptSpecifier id) {
  std::vector<std::string> vals = args.getAllArgValues(id);
  llvm::SmallSet<std::string, 4> uniq(vals.begin(), vals.end());
  if (uniq.size() == 1)
    return *uniq.begin();
  return std::nullopt;
}

// Check the optimization level. Kitsune supports a narrower range of
// optimization levels than clang or flang. If an optimization level is
// specified and it is invalid, return true, otherwise return false.
//
// We currently only support -O1, -O2, -O3, -Og, and -Os. -Oz results in tapir
// loops not being transformed correctly prior to loop spawning. It is not clear
// that -Os and -Oz are relevant for the users of Kitsune. Since -Os seems to
// works, we support it, but if, in the future, Kitsune requires certain passes
// that are not run at -Os, we may remove support for that too.
static bool checkOptLevel(const ArgList &args) {
  if (const Arg *a = args.getLastArg(options::OPT_O_Group)) {
    if (a->getOption().matches(options::OPT_O0)) {
      return true;
    } else if (a->getNumValues()) {
      StringRef v = a->getValue();
      if (v == "1" || v == "2" || v == "3" || v == "g" || v == "s")
        return true;
    }
    return false;
  }
  // An explicit optimization level was not provided. This is ok.
  return true;
}

static void checkThreadsPerBlock(const Arg &a, const ArgList &args,
                                 DiagnosticsEngine &diags) {
  int n = 0;
  StringRef val = a.getValue();
  if (val.empty())
    diags.Report(diag::err_drv_missing_argument) << a.getAsString(args) << 1;
  else if (val.getAsInteger(10, n))
    diags.Report(diag::err_drv_invalid_int_value) << a.getAsString(args) << val;
  else if (n < 1 || n > 1024)
    diags.Report(diag::err_drv_kit_threads_per_block) << a.getAsString(args);
}

void clang::driver::checkKitOptions(const ArgList &args, bool isKitsuneFrontend,
                                    KitDriverMode driverMode, bool isUsingLTO,
                                    StringRef tripleStr,
                                    unsigned amdgpuCodeObjectVersion,
                                    DiagnosticsEngine &diags) {
  llvm::Triple triple = llvm::Triple(tripleStr);

  // If this is not a Kitsune frontend, Kitsune options are not allowed.
  if (!isKitsuneFrontend) {
    for (Arg *a : args.filtered(options::OPT_kitsune_Group)) {
      diags.Report(diag::err_drv_kit_frontend_only) << a->getSpelling();
      return;
    }
  }

  // If this is a Kitsune frontend, some options have a different range of
  // allowed values.
  if (isKitsuneFrontend) {
    if (driverMode == KitDriverMode::Fortran) {
      if (Arg *a = args.getLastArg(options::OPT_ffp_contract)) {
        StringRef fpContract = a->getValue();
        if (fpContract == "on" || fpContract == "fast-honor-pragmas") {
          diags.Report(diag::err_drv_kit_unsupported_optarg)
              << a->getSpelling() << fpContract << llvm::kitFortranFrontend();
          return;
        }
      }
    }

    if (!checkOptLevel(args)) {
      diags.Report(diag::err_drv_kit_bad_opt_level)
          << args.getLastArg(options::OPT_O_Group)->getAsString(args);
      return;
    }
  }

  bool isKokkos = args.hasArg(options::OPT_kokkos);
  bool isKokkosNoInit = args.hasArg(options::OPT_kokkos_no_init);
  if (isKokkos || isKokkosNoInit) {
    auto getKokkosArgSpelling = [](const ArgList &args) -> StringRef {
      return args.getLastArg(options::OPT_kokkos, options::OPT_kokkos_no_init)
          ->getSpelling();
    };

    if constexpr (!llvm::kitKokkosEnabled()) {
      diags.Report(diag::err_drv_kit_kokkos_disabled);
      return;
    }

    // Kokkos-mode is only allowed with the C++ frontend.
    if (driverMode != KitDriverMode::CPlusPlus) {
      diags.Report(diag::err_drv_kit_frontend_badopt)
          << getKokkosArgSpelling(args) << kitCXXFrontend();
      return;
    }

    // If --kokkos is provided, then a tapir target must also be provided.
    if (!args.hasArg(options::OPT_tapir_EQ)) {
      diags.Report(diag::err_drv_kit_tapir_required)
          << getKokkosArgSpelling(args);
      return;
    }
  }

  // If --tapir-plugin= is provided, then a tapir target must also be provided.
  // That target must be 'custom', but that will be checked later.
  if (args.hasArg(options::OPT_tapir_plugin_EQ))
    if (not args.hasArg(options::OPT_tapir_EQ))
      diags.Report(diag::err_drv_kit_plugin_wrong_target);

  // Check that the --tapir flag has a valid value. This stops us from
  // reporting multiple errors because the flag is examined in several places.
  if (const Arg *a = args.getLastArg(options::OPT_tapir_EQ)) {
    std::optional<llvm::TTID> tt = llvm::fromString<llvm::TTID>(a->getValue());
    if (not tt) {
      diags.Report(diag::err_drv_invalid_value)
          << a->getAsString(args) << a->getValue();
      return;
    }

    if (args.hasArg(options::OPT_tapir_plugin_EQ))
      if (*tt != llvm::TTID::Custom)
        diags.Report(diag::err_drv_kit_plugin_wrong_target);

    if (!isEnabledTT(*tt))
      diags.Report(diag::err_drv_kit_target_not_enabled) << llvm::toString(*tt);

    if (*tt == llvm::TTID::Custom) {
      const Arg *a = args.getLastArg(options::OPT_tapir_plugin_EQ);
      if (!a)
        diags.Report(diag::err_drv_kit_plugin_missing);
      if (std::optional<std::string> plugin =
              getUniqueArgValue(args, options::OPT_tapir_plugin_EQ)) {
        if (plugin->empty())
          diags.Report(diag::err_drv_missing_argument)
              << a->getAsString(args) << 1;
      } else {
        diags.Report(diag::err_drv_kit_plugin_multiple);
      }
    } else if (*tt == llvm::TTID::OpenCilk) {
      if (!triple.isOSLinux() && !triple.isOSFreeBSD() && !triple.isMacOSX())
        diags.Report(diag::err_drv_kit_tt_system)
            << llvm::toString(*tt) << triple.getOSName();

      switch (triple.getArch()) {
      case llvm::Triple::x86:
      case llvm::Triple::x86_64:
      case llvm::Triple::arm:
      case llvm::Triple::armeb:
      case llvm::Triple::aarch64:
      case llvm::Triple::aarch64_be:
        break;
      default:
        diags.Report(diag::err_drv_kit_tt_arch)
            << llvm::toString(*tt) << triple.getArchName();
        break;
      }
    } else if (*tt == llvm::TTID::Qthreads) {
      if (!triple.isOSLinux() && !triple.isMacOSX())
        diags.Report(diag::err_drv_kit_tt_system)
            << llvm::toString(*tt) << triple.getOSName();

      switch (triple.getArch()) {
      case llvm::Triple::x86:
      case llvm::Triple::x86_64:
      case llvm::Triple::arm:
      case llvm::Triple::armeb:
      case llvm::Triple::aarch64:
      case llvm::Triple::aarch64_be:
        break;
      default:
        diags.Report(diag::err_drv_kit_tt_arch)
            << llvm::toString(*tt) << triple.getArchName();
        break;
      }
    }

    if (args.hasArg(options::OPT_offload_targets_EQ))
      diags.Report(clang::diag::err_drv_kit_offload);

    // Kitsune does not support ROCm ABI versions < 5. But that should only be
    // relevant when using the Kitsune frontend.
    if (amdgpuCodeObjectVersion < 5)
      diags.Report(diag::err_drv_kit_hip_code_object_version)
          << amdgpuCodeObjectVersion;

    // Kitsune supports a narrower range of optimization levels than clang or
    // flang. If we cannot determine the speedup level, this will issue a
    // diagnostic.
    unsigned speedupLevel = getSpeedupLevel(args, diags);

    // The --tapir option requires optimization level O1 or higher, unless the
    // tapir target is set to nolo. The latter allows -O0 because no lowering
    // takes place and it is very useful to just dump out "tapirized" LLVM IR.
    if (speedupLevel == 0 && *tt != llvm::TTID::Nolo)
      diags.Report(clang::diag::err_drv_kit_optzns_required);

    // The way the middle-end passes are built, the tapir passes are not run if
    // LTO is enabled and the optimization level is < O2. It is not clear why
    // this is the case, but until we decide whether we want to enable tapir
    // lowering at O1 with LTO, don't allow it at all in the frontend. In this
    // case, we don't make an exception for --tapir=nolo
    bool isLTO =
        args.hasArg(options::OPT_flto) || args.hasArg(options::OPT_flto_EQ);
    if (isLTO && speedupLevel < 2)
      diags.Report(clang::diag::err_drv_kit_lto_o2_required);

    // With the cuda tapir target, if debug info is enabled, ptxas cannot be
    // run with optimizations because it does not support "optimized debugging".
    // Just emit a warning so the user is aware of the consequences of using
    // this combination of options.
    if (*tt == llvm::TTID::Cuda && args.getLastArg(options::OPT_g_Group)) {
      if (speedupLevel == 1) {
        diags.Report(clang::diag::warn_drv_kit_cuda_optzns_debug)
            << speedupLevel;
      } else if (speedupLevel > 1) {
        diags.Report(clang::diag::err_drv_kit_cuda_optzns_debug);
        return;
      }
    }
  }

  // Check that the --tapir-cuda-arch option has a valid value. If an empty
  // string is returned, the option has an invalid value.
  if (const Arg *a = args.getLastArg(options::OPT_tapir_cuda_arch_EQ)) {
    OffloadArch arch = StringToOffloadArch(a->getValue());
    if (arch == OffloadArch::UNKNOWN || !IsNVIDIAOffloadArch(arch))
      diags.Report(diag::err_drv_kit_bad_cuda_arch) << a->getValue();
  }

  // Check that the --tapir-cuda-arch option has a valid value. If an empty
  // string is returned, the option has an invalid value.
  if (const Arg *a = args.getLastArg(options::OPT_tapir_hip_arch_EQ)) {
    OffloadArch arch = StringToOffloadArch(a->getValue());
    if (arch == OffloadArch::UNKNOWN || !IsAMDOffloadArch(arch))
      diags.Report(diag::err_drv_kit_bad_hip_arch) << a->getValue();
  }

  // Check that options accepting numeric arguments are within a valid range.
  if (Arg *a = args.getLastArg(options::OPT_tapir_gpu_tpb_EQ))
    checkThreadsPerBlock(*a, args, diags);

  if (Arg *a = args.getLastArg(options::OPT_tapir_gpu_max_tpb_EQ))
    checkThreadsPerBlock(*a, args, diags);

  for (OptSpecifier opt :
       {options::OPT_tapir_hip_sramecc_EQ, options::OPT_tapir_hip_xnack_EQ}) {
    if (const Arg *a = args.getLastArg(opt)) {
      StringRef val = a->getValue();
      if (val.empty())
        diags.Report(diag::err_drv_missing_argument)
            << a->getAsString(args) << 1;
      else if (not llvm::fromString<llvm::MaybeBool>(a->getValue()))
        diags.Report(diag::err_drv_invalid_argument_to_option)
            << val << a->getOption().getName();
    }
  }

  // If LTO is enabled for use with Kitsune, the only linker that can be used is
  // lld built with Kitsune. Using any other linker is not allowed.
  if (isUsingLTO && args.getLastArg(options::OPT_tapir_EQ)) {
    if (const Arg *a =
            args.getLastArg(options::OPT_fuse_ld_EQ, options::OPT_ld_path_EQ))
      diags.Report(diag::err_drv_kit_lto_disallowed_arg) << a->getSpelling();
  }
}

unsigned clang::driver::getSpeedupLevel(const opt::ArgList &args,
                                        DiagnosticsEngine &diags) {
  unsigned defaultSpeedup = KitOptions::defaultSpeedupLevel;
  if (const opt::Arg *a = args.getLastArg(OPT_O_Group)) {
    const opt::Option &option = a->getOption();
    if (option.matches(OPT_O0)) {
      return 0;
    } else if (a->getNumValues()) {
      StringRef v = a->getValue();
      if (v == "1" || v == "g")
        return 1;
      else if (v == "2" || v == "s" || v == "z")
        return 2;
      else if (v == "3")
        return 3;
    }
    llvm_unreachable("getSpeedupLevel: unsupported optimization level");
  }
  return defaultSpeedup;
}

unsigned clang::driver::getSizeLevel(const opt::ArgList &args,
                                     DiagnosticsEngine &diags) {
  const opt::Arg *a = args.getLastArg(OPT_O_Group);
  if (a && a->getNumValues()) {
    StringRef v = a->getValue();
    if (v == "s")
      return 1;
    else if (v == "z")
      return 2;
  }
  return 0;
}

std::optional<TTID> clang::driver::parseTTIfValid(const opt::ArgList &args) {
  if (!args.hasArg(OPT_tapir_EQ))
    return std::nullopt;
  return fromString<TTID>(args.getLastArgValue(OPT_tapir_EQ));
}

std::optional<StringRef>
clang::driver::getTTConfigFileName(const opt::ArgList &args) {
  // This will be called even if the value of the --tapir option is invalid.
  std::optional<TTID> tt = fromString<TTID>(args.getLastArgValue(OPT_tapir_EQ));
  if (!tt)
    return std::nullopt;

  switch (*tt) {
  case TTID::Nolo:
    return "nolo.cfg";
  case TTID::Serial:
    return "serial.cfg";
  case TTID::Cuda:
    return "cuda.cfg";
  case TTID::Custom:
    // The custom tapir target does not use a configuration file.
    return std::nullopt;
  case TTID::Hip:
    return "hip.cfg";
  case TTID::Lambda:
    return "lambda.cfg";
  case TTID::OMPTask:
    return "omptask.cfg";
  case TTID::OpenCilk:
    return "opencilk.cfg";
  case TTID::OpenMP:
    return "openmp.cfg";
  case TTID::Pthreads:
    return "pthreads.cfg";
  case TTID::Qthreads:
    return "qthreads.cfg";
  case TTID::Realm:
    return "realm.cfg";
  }
  llvm_unreachable("getTTConfigFile: TTID not handled");
}

static std::vector<std::string>
parseCommaSeparatedList(StringRef s, DiagnosticsEngine &diags) {
  std::vector<std::string> list;
  std::string tok;
  std::istringstream iss(s.str());
  while (std::getline(iss, tok, ','))
    list.push_back(tok);
  return list;
}

static void parseKitsuneCommonGPUArgs(KitOptions &opts, const ArgList &args,
                                      const OptTable &optTable,
                                      DiagnosticsEngine &diags) {
  if (const Arg *a = args.getLastArg(OPT_tapir_gpu_tpb_EQ)) {
    unsigned n;
    StringRef val = a->getValue();
    val.getAsInteger(10, n);
    opts.setFixedThreadsPerBlock(n);
  }

  if (const Arg *a = args.getLastArg(OPT_tapir_gpu_max_tpb_EQ)) {
    unsigned n;
    StringRef val = a->getValue();
    val.getAsInteger(10, n);
    opts.setMaxThreadsPerBlock(n);
  }

  opts.setGPUPrefetch(args.hasFlag(OPT_tapir_gpu_prefetch,
                                   OPT_tapir_gpu_no_prefetch,
                                   KitOptions::defaultGPUPrefetch));
}

static bool parseKitsuneCudaArgs(KitOptions &opts, const ArgList &args,
                                 const OptTable &optTable,
                                 DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  const OptSpecifier requiredOpts[] = {
      OPT_tapir_cuda_arch_EQ, OPT_tapir_cuda_virt_arch_EQ,
      OPT_tapir_cuda_features_EQ, OPT_tapir_cuda_runtime_bc_EQ};
  for (OptSpecifier opt : requiredOpts)
    if (!args.hasArg(opt))
      diags.Report(diag::err_drv_kit_missing_required)
          << optTable.getOptionName(opt);

  if (diags.getNumErrors() > numErrorsBefore)
    return false;

  opts.setCudaArch(args.getLastArgValue(OPT_tapir_cuda_arch_EQ));
  opts.setCudaVirtArch(args.getLastArgValue(OPT_tapir_cuda_virt_arch_EQ));
  opts.setCudaFeatures(args.getLastArgValue(OPT_tapir_cuda_features_EQ));
  opts.setCudaRuntimeBCFile(args.getLastArgValue(OPT_tapir_cuda_runtime_bc_EQ));

  parseKitsuneCommonGPUArgs(opts, args, optTable, diags);

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneCustomArgs(KitOptions &opts, const ArgList &args,
                                   const OptTable &optTable,
                                   DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  const OptSpecifier requiredOpts[] = {OPT_tapir_plugin_EQ};
  for (OptSpecifier opt : requiredOpts)
    if (!args.hasArg(opt))
      diags.Report(diag::err_drv_kit_missing_required)
          << optTable.getOptionName(opt);

  if (diags.getNumErrors() > numErrorsBefore)
    return false;

  StringRef pluginFile = args.getLastArgValue(OPT_tapir_plugin_EQ);
  Expected<TTPlugin> ttPlugin = TTPlugin::load(pluginFile);
  if (!ttPlugin)
    diags.Report(diag::err_drv_kit_plugin_invalid)
        << toString(ttPlugin.takeError());

  opts.setTTPlugin(pluginFile);

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneHipArgs(KitOptions &opts, const ArgList &args,
                                const OptTable &optTable,
                                DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  const OptSpecifier requiredOpts[] = {
      OPT_tapir_hip_arch_EQ,        OPT_tapir_hip_features_EQ,
      OPT_tapir_hip_runtime_bcs_EQ, OPT_tapir_lld_EQ,
      OPT_tapir_hip_sramecc_EQ,     OPT_tapir_hip_xnack_EQ};
  for (OptSpecifier opt : requiredOpts)
    if (!args.hasArg(opt))
      diags.Report(diag::err_drv_kit_missing_required)
          << optTable.getOptionName(opt);

  if (diags.getNumErrors() > numErrorsBefore)
    return false;

  opts.setHipArch(args.getLastArgValue(OPT_tapir_hip_arch_EQ));
  opts.setHipFeatures(args.getLastArgValue(OPT_tapir_hip_features_EQ));
  for (StringRef file : parseCommaSeparatedList(
           args.getLastArgValue(OPT_tapir_hip_runtime_bcs_EQ), diags))
    opts.addHipRuntimeBCFile(file);
  opts.setLLD(args.getLastArgValue(OPT_tapir_lld_EQ));

  if (const Arg *a = args.getLastArg(OPT_tapir_hip_sramecc_EQ)) {
    StringRef val = a->getValue();
    if (std::optional<llvm::MaybeBool> mb =
            llvm::fromString<llvm::MaybeBool>(val))
      opts.setHipSramECC(*mb);
    else
      diags.Report(diag::err_drv_invalid_argument_to_option)
          << val << a->getOption().getName();
  }

  if (const Arg *a = args.getLastArg(OPT_tapir_hip_xnack_EQ)) {
    StringRef val = a->getValue();
    if (std::optional<llvm::MaybeBool> mb =
            llvm::fromString<llvm::MaybeBool>(val))
      opts.setHipXnack(*mb);
    else
      diags.Report(diag::err_drv_invalid_argument_to_option)
          << val << a->getOption().getName();
  }

  parseKitsuneCommonGPUArgs(opts, args, optTable, diags);

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneLambdaArgs(KitOptions &opts, const ArgList &args,
                                   const OptTable &optTable,
                                   DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // Don't hit unreachable if an error has already occurred
  if (!numErrorsBefore)
    llvm_unreachable("NOT IMPLEMENTED: ParseKitsuneLambdaArgs");

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneOMPTaskArgs(KitOptions &opts, const ArgList &args,
                                    const OptTable &optTable,
                                    DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // Don't hit unreachable if an error has already occurred
  if (!numErrorsBefore)
    llvm_unreachable("NOT IMPLEMENTED: ParseKitsuneOMPTaskArgs");

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneOpenCilkArgs(KitOptions &opts, const ArgList &args,
                                     const OptTable &optTable,
                                     DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  for (OptSpecifier opt : {OPT_tapir_opencilk_runtime_bc_EQ})
    if (!args.hasArg(opt))
      diags.Report(diag::err_drv_kit_missing_required)
          << optTable.getOptionName(opt);

  if (diags.getNumErrors() > numErrorsBefore)
    return false;

  opts.setOpenCilkRuntimeBCFile(
      args.getLastArgValue(OPT_tapir_opencilk_runtime_bc_EQ));

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneRealmArgs(KitOptions &opts, const ArgList &args,
                                  const OptTable &optTable,
                                  DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // Don't hit unreachable if an error has already occurred
  if (!numErrorsBefore)
    llvm_unreachable("NOT IMPLEMENTED: ParseKitsuneRealmArgs");

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneTTArgs(KitOptions &kitOpts, TTID tt,
                               const ArgList &args, const OptTable &optTable,
                               DiagnosticsEngine &diags) {
  switch (tt) {
  case llvm::TTID::Nolo:
    return true;
  case llvm::TTID::Cuda:
    return parseKitsuneCudaArgs(kitOpts, args, optTable, diags);
  case llvm::TTID::Custom:
    return parseKitsuneCustomArgs(kitOpts, args, optTable, diags);
  case llvm::TTID::Hip:
    return parseKitsuneHipArgs(kitOpts, args, optTable, diags);
  case llvm::TTID::Lambda:
    return parseKitsuneLambdaArgs(kitOpts, args, optTable, diags);
  case llvm::TTID::OMPTask:
    return parseKitsuneOMPTaskArgs(kitOpts, args, optTable, diags);
  case llvm::TTID::OpenCilk:
    return parseKitsuneOpenCilkArgs(kitOpts, args, optTable, diags);
  case llvm::TTID::OpenMP:
  case llvm::TTID::Pthreads:
  case llvm::TTID::Qthreads:
    return true;
  case llvm::TTID::Realm:
    return parseKitsuneRealmArgs(kitOpts, args, optTable, diags);
  case llvm::TTID::Serial:
    return true;
  }
  llvm_unreachable("parseKitsuneTTArgs: TTID not handled");
}

bool clang::driver::parseKitsuneArgs(KitOptions &kitOpts, const char *argv0,
                                     const ArgList &args,
                                     const OptTable &optTable,
                                     DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  kitOpts.setKitsuneFrontend(isKitsuneFrontend(argv0));
  kitOpts.setStripmineLoops(args.hasArg(OPT_fstripmine));

  if (const Arg *arg = args.getLastArg(OPT_tapir_EQ)) {
    if (std::optional<llvm::TTID> tt = fromString<TTID>(arg->getValue())) {
      parseKitsuneTTArgs(kitOpts, *tt, args, optTable, diags);
      kitOpts.setTTID(*tt);
    }
  }

  kitOpts.setKokkos(args.hasArg(OPT_kokkos));
  kitOpts.setKokkosNoInit(args.hasArg(OPT_kokkos_no_init));

  return diags.getNumErrors() == numErrorsBefore;
}
