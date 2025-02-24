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
  if (const opt::Arg *A = Args.getLastArg(options::OPT_ftapir_EQ))
    return parseTapirTarget(*A);
  return std::nullopt;
}

StringRef clang::parseTapirCudaArch(const opt::Arg &A) {
  StringRef sm = A.getValue();
  if (sm == "sm_50" || sm == "sm_52" || sm == "sm_53" || sm == "sm_60" ||
      sm == "sm_62" || sm == "sm_70" || sm == "sm_72" || sm == "sm_75" ||
      sm == "sm_80" || sm == "sm_86" || sm == "sm_90")
    return A.getValue();
  return "";
}

std::optional<std::string> clang::parseTapirCudaArch(const opt::ArgList &Args) {
  if (const opt::Arg *A = Args.getLastArg(options::OPT_ftapir_cuda_arch_EQ))
    return parseTapirCudaArch(*A).str();
  return std::nullopt;
}

StringRef clang::parseTapirHipArch(const opt::Arg &A) {
  StringRef v = A.getValue();
  if (v == "gfx906" || v == "gfx908" || v == "gfx90a" || v == "gfx90c" ||
      v == "gfx942")
    return A.getValue();
  return "";
}

std::optional<std::string> clang::parseTapirHipArch(const opt::ArgList &Args) {
  if (const opt::Arg *A = Args.getLastArg(options::OPT_ftapir_hip_arch_EQ))
    return parseTapirHipArch(*A).str();
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
      return std::nullopt;
    }
  }
  return std::nullopt;
}
