//===--- KitsuneOptions.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements initialization of KitsuneOptions from the -cc1/-fc1
// command line.
//
//===----------------------------------------------------------------------===//

// NOTE: This is currently in clang/lib/Basic because it is shared between
// clang and flang. But flang would like to remove the dependence on clangBasic.
// The main reason this cannot be moved to llvm/lib/Frontend/Driver to live
// alongside KitsuneOptions.h is the DiagnosticsEngine object which is defined
// in clangBasic. If the LLVM folks upstream move the DiagnosticsEngine where
// it can be shared without needing to resort to clangBasic, we should be able
// to move this file as well.

#include "llvm/Frontend/Driver/KitsuneOptions.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tapir.h"
#include "llvm/Frontend/Tapir/CommandLine.h"
#include "llvm/Option/ArgList.h"

#include <sstream>
#include <vector>

using namespace clang;
using namespace clang::driver::options;
using namespace llvm::opt;

using KitsuneOptions = llvm::driver::KitsuneOptions;

static std::vector<std::string>
parseCommaSeparatedList(StringRef s, DiagnosticsEngine &diags) {
  std::vector<std::string> list;
  std::string tok;
  std::istringstream iss(s.str());
  while (std::getline(iss, tok, ','))
    list.push_back(tok);
  return list;
}

static void parseKitsuneCommonGPUArgs(KitsuneOptions &opts, const ArgList &args,
                                      const OptTable &optTable,
                                      DiagnosticsEngine &diags) {
  if (const Arg *a = args.getLastArg(OPT_tapir_threads_per_block_EQ)) {
    unsigned n;
    StringRef val = a->getValue();
    val.getAsInteger(10, n);
    opts.setFixedThreadsPerBlock(n);
  }

  if (const Arg *a = args.getLastArg(OPT_tapir_max_threads_per_block_EQ)) {
    unsigned n;
    StringRef val = a->getValue();
    val.getAsInteger(10, n);
    opts.setMaxThreadsPerBlock(n);
  }

  opts.setGPUPrefetch(args.hasFlag(OPT_tapir_gpu_prefetch,
                                   OPT_tapir_gpu_no_prefetch,
                                   KitsuneOptions::defaultGPUPrefetch));
}

static bool parseKitsuneCudaArgs(KitsuneOptions &opts, const ArgList &args,
                                 const OptTable &optTable,
                                 DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  const OptSpecifier requiredOpts[] = {
      OPT_tapir_cuda_arch_EQ, OPT_tapir_cuda_virt_arch_EQ,
      OPT_tapir_cuda_features_EQ, OPT_tapir_cuda_runtime_bc_EQ};
  for (OptSpecifier opt : requiredOpts)
    if (!args.hasArg(opt))
      diags.Report(diag::err_drv_kitsune_missing_required)
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

static bool parseKitsuneHipArgs(KitsuneOptions &opts, const ArgList &args,
                                const OptTable &optTable,
                                DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  const OptSpecifier requiredOpts[] = {
      OPT_tapir_hip_arch_EQ,        OPT_tapir_hip_features_EQ,
      OPT_tapir_hip_runtime_bcs_EQ, OPT_tapir_lld_EQ,
      OPT_tapir_hip_sramecc_EQ,     OPT_tapir_hip_xnack_EQ};
  for (OptSpecifier opt : requiredOpts)
    if (!args.hasArg(opt))
      diags.Report(diag::err_drv_kitsune_missing_required)
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
    if (llvm::ErrorOr<llvm::MaybeBool> parsed = llvm::parseMaybeBool(val))
      opts.setHipSramECC(parsed.get());
    else
      diags.Report(diag::err_drv_invalid_argument_to_option)
          << val << a->getOption().getName();
  }

  if (const Arg *a = args.getLastArg(OPT_tapir_hip_xnack_EQ)) {
    StringRef val = a->getValue();
    if (llvm::ErrorOr<llvm::MaybeBool> parsed = llvm::parseMaybeBool(val))
      opts.setHipXnack(parsed.get());
    else
      diags.Report(diag::err_drv_invalid_argument_to_option)
          << val << a->getOption().getName();
  }

  parseKitsuneCommonGPUArgs(opts, args, optTable, diags);

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneLambdaArgs(KitsuneOptions &opts, const ArgList &args,
                                   const OptTable &optTable,
                                   DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // Don't hit unreachable if an error has already occurred
  if (!numErrorsBefore)
    llvm_unreachable("NOT IMPLEMENTED: ParseKitsuneLambdaargs");

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneOMPTaskArgs(KitsuneOptions &opts, const ArgList &args,
                                    const OptTable &optTable,
                                    DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // Don't hit unreachable if an error has already occurred
  if (!numErrorsBefore)
    llvm_unreachable("NOT IMPLEMENTED: ParseKitsuneOMPTaskargs");

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneOpenCilkArgs(KitsuneOptions &opts, const ArgList &args,
                                     const OptTable &optTable,
                                     DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  for (OptSpecifier opt : {OPT_tapir_opencilk_runtime_bc_EQ})
    if (!args.hasArg(opt))
      diags.Report(diag::err_drv_kitsune_missing_required)
          << optTable.getOptionName(opt);

  if (diags.getNumErrors() > numErrorsBefore)
    return false;

  opts.setOpenCilkRuntimeBCFile(
      args.getLastArgValue(OPT_tapir_opencilk_runtime_bc_EQ));

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneOpenMPArgs(KitsuneOptions &opts, const ArgList &args,
                                   const OptTable &optTable,
                                   DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // Don't hit unreachable if an error has already occurred
  if (!numErrorsBefore)
    llvm_unreachable("NOT IMPLEMENTED: ParseKitsuneOpenMPargs");

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneQthreadsArgs(KitsuneOptions &opts, const ArgList &args,
                                     const OptTable &optTable,
                                     DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // Don't hit unreachable if an error has already occurred
  if (!numErrorsBefore)
    llvm_unreachable("NOT IMPLEMENTED: ParseKitsuneQthreadsargs");

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneRealmArgs(KitsuneOptions &opts, const ArgList &args,
                                  const OptTable &optTable,
                                  DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // Don't hit unreachable if an error has already occurred
  if (!numErrorsBefore)
    llvm_unreachable("NOT IMPLEMENTED: ParseKitsuneRealmargs");

  return diags.getNumErrors() == numErrorsBefore;
}

bool llvm::driver::KitsuneOptions::parseArgsInto(const char *argv0,
                                                 const ArgList &args,
                                                 const OptTable &optTable,
                                                 DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  setKitsuneFrontend(clang::driver::IsKitsuneFrontend(argv0));
  setStripmineLoops(args.hasArg(OPT_fstripmine));

  setTapirVerbose(args.hasArg(OPT_tapir_verbose));
  setKitrtVerbose(args.hasArg(OPT_kitrt_verbose));

  if (std::optional<llvm::TapirTargetID> tt = parseTapirTargetIfValid(args)) {
    switch (*tt) {
    case llvm::TapirTargetID::None:
      break;
    case llvm::TapirTargetID::Cuda:
      parseKitsuneCudaArgs(*this, args, optTable, diags);
      break;
    case llvm::TapirTargetID::Hip:
      parseKitsuneHipArgs(*this, args, optTable, diags);
      break;
    case llvm::TapirTargetID::Lambda:
      parseKitsuneLambdaArgs(*this, args, optTable, diags);
      break;
    case llvm::TapirTargetID::OMPTask:
      parseKitsuneOMPTaskArgs(*this, args, optTable, diags);
      break;
    case llvm::TapirTargetID::OpenCilk:
      parseKitsuneOpenCilkArgs(*this, args, optTable, diags);
      break;
    case llvm::TapirTargetID::OpenMP:
      parseKitsuneOpenMPArgs(*this, args, optTable, diags);
      break;
    case llvm::TapirTargetID::Qthreads:
      parseKitsuneQthreadsArgs(*this, args, optTable, diags);
      break;
    case llvm::TapirTargetID::Realm:
      parseKitsuneRealmArgs(*this, args, optTable, diags);
      break;
    case llvm::TapirTargetID::Serial:
      break;
    default:
      llvm_unreachable("ParseKitsuneargs: TapirTargetID not handled");
    }
    setTapirTarget(*tt);
  }

  setKokkos(args.hasArg(OPT_kokkos));
  setKokkosNoInit(args.hasArg(OPT_kokkos_no_init));

  return diags.getNumErrors() == numErrorsBefore;
}
