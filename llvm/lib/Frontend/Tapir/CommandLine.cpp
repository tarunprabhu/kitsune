//===- CommandLine.cpp ---------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Helper functions to parse tapir-specific command line options
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/Tapir/CommandLine.h"
#include "llvm/ADT/StringSwitch.h"

namespace llvm {

ErrorOr<TapirTargetID> parseTapirTarget(StringRef s) {
  return StringSwitch<ErrorOr<TapirTargetID>>(s)
      .Case("none", TapirTargetID::None)
      .Case("cuda", TapirTargetID::Cuda)
      .Case("hip", TapirTargetID::Hip)
      .Case("lambda", TapirTargetID::Lambda)
      .Case("omptask", TapirTargetID::OMPTask)
      .Case("opencilk", TapirTargetID::OpenCilk)
      .Case("openmp", TapirTargetID::OpenMP)
      .Case("qthreads", TapirTargetID::Qthreads)
      .Case("realm", TapirTargetID::Realm)
      .Case("serial", TapirTargetID::Serial)
      .Default(std::make_error_code(std::errc::invalid_argument));
}

ErrorOr<MaybeBool> parseMaybeBool(StringRef s) {
  return StringSwitch<ErrorOr<MaybeBool>>(s)
      .Case("off", MaybeBool::Off)
      .Case("on", MaybeBool::On)
      .Case("any", MaybeBool::Any)
      .Default(std::make_error_code(std::errc::invalid_argument));
}

} // namespace llvm
