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
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Transforms/Tapir/TapirTargetIDs.h"

using namespace clang::driver;
using namespace llvm;

std::optional<TapirTargetID> clang::parseTapirTarget(const opt::Arg &A) {
  return llvm::parseTapirTarget(A.getValue());
}

std::optional<TapirTargetID> clang::parseTapirTarget(const opt::ArgList &Args) {
  if (const opt::Arg *A = Args.getLastArg(options::OPT_tapir_EQ))
    return parseTapirTarget(*A);
  return std::nullopt;
}

std::optional<StringRef>
clang::getTargetConfigFileName(const opt::ArgList &Args) {
  if (std::optional<TapirTargetID> tt = parseTapirTarget(Args)) {
    switch (*tt) {
    case TapirTargetID::None:
      return "none.cfg";
    case TapirTargetID::Serial:
      return "serial.cfg";
    case TapirTargetID::Cuda:
      return "cuda.cfg";
    case TapirTargetID::Hip:
      return "hip.cfg";
    case TapirTargetID::OpenCilk:
      return "opencilk.cfg";
    case TapirTargetID::OpenMP:
      return "openmp.cfg";
    case TapirTargetID::Qthreads:
      return "qthreads.cfg";
    case TapirTargetID::Realm:
      return "realm.cfg";
    default:
      llvm_unreachable("getTargetConfigFile: TapirTargetID not handled");
    }
  }
  return std::nullopt;
}
