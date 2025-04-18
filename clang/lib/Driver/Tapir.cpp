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
#include "clang/Driver/Options.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Frontend/Tapir/CommandLine.h"
#include "llvm/Frontend/Tapir/Tapir.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"

using namespace clang::driver;
using namespace llvm;

namespace clang {

std::optional<TapirTargetID> parseTapirTargetIfValid(const opt::ArgList &Args) {
  if (!Args.hasArg(options::OPT_tapir_EQ))
    return std::nullopt;

  ErrorOr<TapirTargetID> TT =
      parseTapirTarget(Args.getLastArgValue(options::OPT_tapir_EQ));
  if (!TT)
    return std::nullopt;

  return *TT;
}

std::optional<StringRef>
getTapirTargetConfigFileName(const opt::ArgList &Args) {
  if (!Args.hasArg(options::OPT_tapir_EQ))
    return std::nullopt;

  // Even if the value of the --tapir option is invalid, this will get called.
  ErrorOr<TapirTargetID> TT =
      parseTapirTarget(Args.getLastArgValue(options::OPT_tapir_EQ));
  if (!TT)
    return std::nullopt;

  switch (*TT) {
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

} // namespace clang
