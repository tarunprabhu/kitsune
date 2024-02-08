//===--- Tapir.cpp - C Language Family Language Options ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the functions from Tapir.h
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Tapir.h"
#include "clang/Driver/OptionUtils.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Frontend/Driver/KitsuneOptions.h"
#include "llvm/Frontend/Tapir/CommandLine.h"
#include "llvm/Frontend/Tapir/Tapir.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"

using namespace clang::driver;
using namespace llvm;

std::optional<TapirTargetID>
clang::parseTapirTargetIfValid(const opt::ArgList &args) {
  if (!args.hasArg(options::OPT_tapir_EQ))
    return std::nullopt;

  ErrorOr<TapirTargetID> tt =
      parseTapirTarget(args.getLastArgValue(options::OPT_tapir_EQ));
  if (!tt)
    return std::nullopt;

  return *tt;
}

std::optional<StringRef>
clang::getTapirTargetConfigFileName(const opt::ArgList &args) {
  if (!args.hasArg(options::OPT_tapir_EQ))
    return std::nullopt;

  // Even if the value of the --tapir option is invalid, this will get called.
  ErrorOr<TapirTargetID> tt =
      parseTapirTarget(args.getLastArgValue(options::OPT_tapir_EQ));
  if (!tt)
    return std::nullopt;

  switch (*tt) {
  case TapirTargetID::None:
    return "none.cfg";
  case TapirTargetID::Serial:
    return "serial.cfg";
  case TapirTargetID::Cuda:
    return "cuda.cfg";
  case TapirTargetID::Hip:
    return "hip.cfg";
  case TapirTargetID::Lambda:
    return "lambda.cfg";
  case TapirTargetID::OMPTask:
    return "omptask.cfg";
  case TapirTargetID::OpenCilk:
    return "opencilk.cfg";
  case TapirTargetID::OpenMP:
    return "openmp.cfg";
  case TapirTargetID::Qthreads:
    return "qthreads.cfg";
  case TapirTargetID::Realm:
    return "realm.cfg";
  default:
    llvm_unreachable("getTapirTargetConfigFile: TapirTargetID not handled");
  }
}

unsigned clang::getSpeedupLevelAsInt(const opt::ArgList &args,
                                     DiagnosticsEngine &diags) {
  unsigned defaultSpeedup = llvm::driver::KitsuneOptions::defaultSpeedupLevel;
  if (opt::Arg *a = args.getLastArg(options::OPT_O_Group)) {
    if (a->getOption().matches(options::OPT_O0))
      return 0;

    if (a->getOption().matches(options::OPT_Ofast))
      return 3;

    assert(a->getOption().matches(options::OPT_O));

    StringRef val = a->getValue();
    if (val == "s" || val == "z")
      return 2;

    if (val == "g")
      return 1;

    return getLastArgIntValue(args, options::OPT_O, defaultSpeedup, diags);
  }
  return defaultSpeedup;
}
