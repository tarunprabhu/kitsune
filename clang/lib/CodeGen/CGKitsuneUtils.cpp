//===- CGKitsuneUtils.cpp - Utilities used in Kitsune's clang codegen -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities used in Kitsune's clang CodeGen
//
//===----------------------------------------------------------------------===//

#include "CGKitsuneUtils.h"
#include "clang/AST/Attr.h"

using namespace clang;

llvm::TTID clang::CodeGen::getTapirTarget(llvm::ArrayRef<const Attr *> attrs,
                                          llvm::TTID primaryTT) {
  // The TTAttr attribute is guaranteed to appear at most once, so it is safe
  // to return immediately when it is encountered.
  for (const Attr *attr : attrs) {
    if (const auto *ttAttr = dyn_cast<TTAttr>(attr)) {
      switch (ttAttr->getTT()) {
      case TTAttr::Nolo:
        return llvm::TTID::Nolo;
      case TTAttr::Cuda:
        return llvm::TTID::Cuda;
      case TTAttr::Hip:
        return llvm::TTID::Hip;
      case TTAttr::OpenCilk:
        return llvm::TTID::OpenCilk;
      case TTAttr::OpenMP:
        return llvm::TTID::OpenMP;
      case TTAttr::Pthreads:
        return llvm::TTID::Pthreads;
      case TTAttr::Qthreads:
        return llvm::TTID::Qthreads;
      case TTAttr::Serial:
        return llvm::TTID::Serial;
      case TTAttr::Custom:
        llvm_unreachable("Value of tapir target attribute cannot be 'custom'");
      }
      llvm_unreachable("getTapirTarget: TTAttr not handled");
    }
  }
  return primaryTT;
}

unsigned
clang::CodeGen::getKitsuneLaunchAttr(llvm::ArrayRef<const Attr *> attrs) {
  // The KitsuneLaunch attribute is guaranteed to appear at most once, so it is
  // safe to return immediately when it is encountered.
  for (const Attr *attr : attrs)
    if (const auto *launchAttr = dyn_cast<KitsuneLaunchAttr>(attr))
      return launchAttr->getThreadsPerBlock();
  return 0;
}
