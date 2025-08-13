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
#include "llvm/ADT/SetVector.h"
#include "llvm/Object/ObjectFile.h"

using namespace llvm;
using namespace llvm::object;

Expected<bool> llvm::object::hasEmbDeviceCode(const ObjectFile &objFile) {
  for (SectionRef sec : objFile.sections())
    if (Expected<StringRef> name = sec.getName())
      if (*name == KITSUNE_CUDA_CODE_SECTION or
          *name == KITSUNE_HIP_CODE_SECTION)
        return true;
  return false;
}

Expected<SmallVector<TTID, 0>>
llvm::object::getEmbDeviceCodeTTIDs(const ObjectFile &objFile) {
  SmallSetVector<TTID, 2> tts;
  for (SectionRef sec : objFile.sections()) {
    if (Expected<StringRef> name = sec.getName()) {
      if (*name == KITSUNE_CUDA_CODE_SECTION)
        tts.insert(TTID::Cuda);
      else if (*name == KITSUNE_HIP_CODE_SECTION)
        tts.insert(TTID::Hip);
    }
  }
  return tts.takeVector();
}
