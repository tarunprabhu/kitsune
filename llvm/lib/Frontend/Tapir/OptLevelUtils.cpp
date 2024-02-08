//===- CommandLine.cpp ---------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Helper functions to deal with optimization levels which tend to be passed
/// around as a number of different enums and even raw integer values.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/Tapir/OptLevelUtils.h"
#include "llvm/Support/Error.h"

using namespace llvm;

// NOTE: See llvm/lib/Passes/OptimizationLevel.cpp for an explanation for why
// these are defined here.
const OptimizationLevel OptimizationLevel::O0 = {
    /*SpeedLevel*/ 0,
    /*SizeLevel*/ 0};
const OptimizationLevel OptimizationLevel::O1 = {
    /*SpeedLevel*/ 1,
    /*SizeLevel*/ 0};
const OptimizationLevel OptimizationLevel::O2 = {
    /*SpeedLevel*/ 2,
    /*SizeLevel*/ 0};
const OptimizationLevel OptimizationLevel::O3 = {
    /*SpeedLevel*/ 3,
    /*SizeLevel*/ 0};
const OptimizationLevel OptimizationLevel::Os = {
    /*SpeedLevel*/ 2,
    /*SizeLevel*/ 1};
const OptimizationLevel OptimizationLevel::Oz = {
    /*SpeedLevel*/ 2,
    /*SizeLevel*/ 2};

CodeGenOptLevel llvm::mapToCodeGenOptLevel(OptimizationLevel OptLevel) {
  switch (OptLevel.getSpeedupLevel()) {
  case 0:
    return CodeGenOptLevel::None;
  case 1:
    return CodeGenOptLevel::Less;
  case 2:
    return CodeGenOptLevel::Default;
  case 3:
    return CodeGenOptLevel::Aggressive;
  default:
    llvm_unreachable("mapToCodeGenOptLevel: unknown speedup level");
  }
}

OptimizationLevel llvm::mapToOptimizationLevel(unsigned OptLevel) {
  switch (OptLevel) {
  case 0:
    return OptimizationLevel::O0;
  case 1:
    return OptimizationLevel::O1;
  case 2:
    return OptimizationLevel::O2;
  case 3:
    return OptimizationLevel::O3;
  default:
    llvm_unreachable("mapToOptimizationLevel: invalid optimization level");
  }
}
