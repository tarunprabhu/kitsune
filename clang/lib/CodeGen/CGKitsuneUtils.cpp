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

llvm::TapirSpawnStrategy
clang::CodeGen::getTapirSpawnStrategy(llvm::ArrayRef<const Attr *> attrs,
                                      llvm::TTID primaryTT) {
  for (const Attr *attr : attrs) {
    if (const auto *strategyAttr = dyn_cast<TapirStrategyAttr>(attr)) {
      switch (strategyAttr->getTapirStrategyType()) {
      case TapirStrategyAttr::Basic:
        return llvm::TapirSpawnStrategy::Basic;
      case TapirStrategyAttr::DivideAndConquer:
        return llvm::TapirSpawnStrategy::DivideAndConquer;
      case TapirStrategyAttr::GPU:
        return llvm::TapirSpawnStrategy::GPU;
      case TapirStrategyAttr::Sequential:
        return llvm::TapirSpawnStrategy::Sequential;
      }
      llvm_unreachable("getTapirStrategy: TapirStrategyAttr not handled");
    }
  }

  switch (getTapirTarget(attrs, primaryTT)) {
  case llvm::TTID::Nolo:
  case llvm::TTID::Serial:
    return llvm::TapirSpawnStrategy::Sequential;
  case llvm::TTID::Cuda:
  case llvm::TTID::Hip:
    return llvm::TapirSpawnStrategy::GPU;
  case llvm::TTID::OpenCilk:
    return llvm::TapirSpawnStrategy::DivideAndConquer;
  case llvm::TTID::Custom:
  case llvm::TTID::OpenMP:
  case llvm::TTID::Pthreads:
  case llvm::TTID::Qthreads:
    return llvm::TapirSpawnStrategy::Basic;
  default:
    llvm_unreachable("getTapirStrategy: TTID not handled");
  }
}

llvm::TTID clang::CodeGen::getTapirTarget(llvm::ArrayRef<const Attr *> attrs,
                                          llvm::TTID defawlt) {
  // FIXME KITSUNE: This will check for the first occurrence of the tapir target
  // attribute and break immediately if it finds it. Is this what we actually
  // want?
  for (const Attr *attr : attrs) {
    if (const auto *ttAttr = dyn_cast<TapirTargetAttr>(attr)) {
      switch (ttAttr->getTapirTargetAttrType()) {
      case TapirTargetAttr::Nolo:
        return llvm::TTID::Nolo;
      case TapirTargetAttr::Cuda:
        return llvm::TTID::Cuda;
      case TapirTargetAttr::Hip:
        return llvm::TTID::Hip;
      case TapirTargetAttr::OpenCilk:
        return llvm::TTID::OpenCilk;
      case TapirTargetAttr::OpenMP:
        return llvm::TTID::OpenMP;
      case TapirTargetAttr::Pthreads:
        return llvm::TTID::Pthreads;
      case TapirTargetAttr::Qthreads:
        return llvm::TTID::Qthreads;
      case TapirTargetAttr::Serial:
        return llvm::TTID::Serial;
      case TapirTargetAttr::Custom:
        llvm_unreachable("Value of tapir target attribute cannot be 'custom'");
      }
      llvm_unreachable("getTapirTargetAttr: TTID not handled");
    }
  }
  return defawlt;
}

unsigned
clang::CodeGen::getKitsuneLaunchAttr(llvm::ArrayRef<const Attr *> attrs) {
  for (const auto *attr : attrs)
    if (attr->getKind() == attr::KitsuneLaunch)
      return cast<const KitsuneLaunchAttr>(attr)->getThreadsPerBlock();
  return 0;
}
