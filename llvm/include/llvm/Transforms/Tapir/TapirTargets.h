//===--TapirTargets.h - Helper header for tapir targets -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is simply a header file that includes headers for the known tapir
// targets to make it more convenient to include where needed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TAPIR_TARGETS_H
#define LLVM_TAPIR_TARGETS_H

#include "llvm/Transforms/Tapir/CudaABI.h"
#include "llvm/Transforms/Tapir/HipABI.h"
#include "llvm/Transforms/Tapir/LambdaABI.h"
#include "llvm/Transforms/Tapir/OMPTaskABI.h"
#include "llvm/Transforms/Tapir/OpenCilkABI.h"
#include "llvm/Transforms/Tapir/QthreadsABI.h"
#include "llvm/Transforms/Tapir/RealmABI.h"
#include "llvm/Transforms/Tapir/SerialABI.h"

#endif // LLVM_TAPIR_TARGETS_H
