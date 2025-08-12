//===- ObjectUtils.cpp - Utilities for LLVM's object files ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM's object files.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Object/ObjectUtils.h"
#include "kitsune/Config/config.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Object/ObjectFile.h"

using namespace llvm;
using namespace llvm::object;

static bool isDeviceCodeSection(SectionRef sec, TTID tt) {
  if (Expected<StringRef> name = sec.getName())
    return StringSwitch<bool>(*name)
        .Case(KITSUNE_CUDA_CODE_SECTION, tt == TTID::Cuda)
        .Case(KITSUNE_HIP_CODE_SECTION, tt == TTID::Hip)
        .Default(false);
  return false;
}

Expected<bool> llvm::object::hasEmbDeviceCode(const ObjectFile &objFile,
                                              TTID tt) {
  for (SectionRef sec : objFile.sections())
    if (isDeviceCodeSection(sec, tt))
      return true;
  return false;
}
