//===- Tapir.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements common functionality for libLLVMTapirCommon.a which is
// shared with clang, flang and lld.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/TapirTargetIDs.h"

namespace llvm {

std::optional<TapirTargetID> parseTapirTarget(StringRef s) {
  return StringSwitch<std::optional<TapirTargetID>>(s)
      .Case("none", TapirTargetID::None)
      .Case("serial", TapirTargetID::Serial)
      .Case("cuda", TapirTargetID::Cuda)
      .Case("hip", TapirTargetID::Hip)
      .Case("opencilk", TapirTargetID::OpenCilk)
      .Case("openmp", TapirTargetID::OpenMP)
      .Case("qthreads", TapirTargetID::Qthreads)
      .Case("realm", TapirTargetID::Realm)
      .Default(std::nullopt);
}

raw_ostream &operator<<(raw_ostream &os, const TapirTargetID &Target) {
  switch (Target) {
  case TapirTargetID::None:
    return os << "none";
  case TapirTargetID::Serial:
    return os << "serial";
  case TapirTargetID::Cuda:
    return os << "cuda";
  case TapirTargetID::Hip:
    return os << "hip";
  case TapirTargetID::Lambda:
    return os << "lambda";
  case TapirTargetID::OMPTask:
    return os << "omptask";
  case TapirTargetID::OpenCilk:
    return os << "opencilk";
  case TapirTargetID::OpenMP:
    return os << "openmp";
  case TapirTargetID::Qthreads:
    return os << "qthreads";
  case TapirTargetID::Realm:
    return os << "realm";
  case TapirTargetID::Last_TapirTargetID:
    return os << "<invalid>";
  }
  // We don't put this in a default block in the switch above because it results
  // in compiler warnings about default blocks in a switch where all enumeration
  // values are handled. But we want this error in case a new tapir target is
  // added, but this code is not updated.
  llvm_unreachable("Tapir target not handled");
}

} // namespace llvm
