//===- TapirCommandLineUtils.cpp ------------------------------ -*- C++ -*-===//
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

#include "llvm/Transforms/Tapir/TapirCommandLineUtils.h"
#include "llvm/ADT/StringSwitch.h"

namespace llvm {

std::optional<TapirTargetID> parseTapirTarget(StringRef s) {
  return StringSwitch<std::optional<TapirTargetID>>(s)
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
      .Default(std::nullopt);
}

ErrorOr<std::optional<bool>> parseOptionalBool(StringRef s) {
  return StringSwitch<ErrorOr<std::optional<bool>>>(s)
      .Case("off", false)
      .Case("on", true)
      .Case("any", std::nullopt)
      .Default(std::make_error_code(std::errc::invalid_argument));
}

} // namespace llvm
