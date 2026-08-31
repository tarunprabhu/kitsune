//===- Passes.h - Headers with Kitsune's passes -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is just a convenience header that includes the headers for all Kitsune's
// passes. It is mainly intended to be included llvm/lib/Passes/PassBuilder.cpp.
// This way, whenever a new Kitsune-specific pass is added, we don't have to
// update a file in llvm/, we only have to add it "locally" here. This reduces
// Kitsune's footprint in core LLVM, something we actively aim for.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_PASSES_PASSES_H
#define KITSUNE_PASSES_PASSES_H

#include "kitsune/Analysis/EarlyVerification.h"
#include "kitsune/Analysis/PreLowerVerification.h"
#include "kitsune/Analysis/TTObjectsAnalysis.h"
#include "kitsune/Analysis/TTOptionsPrinter.h"
#include "kitsune/CodeGen/CodeGenFatBinaries.h"
#include "kitsune/CodeGen/EmbLowerKitIntrinsics.h"
#include "kitsune/CodeGen/LowerIntrinsics.h"
#include "kitsune/CodeGen/StripKitAddrSpaces.h"
#include "kitsune/Transforms/DeLICM.h"
#include "kitsune/Transforms/EarlyAnnotate.h"
#include "kitsune/Transforms/EmbLinkLibDeviceBitcode.h"
#include "kitsune/Transforms/EmbLowerIntrinsicsEarly.h"
#include "kitsune/Transforms/EmbLowerWarpIntrinsics.h"
#include "kitsune/Transforms/EmbOptimize.h"
#include "kitsune/Transforms/EmbPrepare.h"
#include "kitsune/Transforms/EmbResolveLibDeviceCalls.h"
#include "kitsune/Transforms/GenerateCtors.h"
#include "kitsune/Transforms/HoistAllocas.h"
#include "kitsune/Transforms/Instrument.h"
#include "kitsune/Transforms/LowerReduceIntrinsics.h"
#include "kitsune/Transforms/NormalizeLoopControlBlocks.h"
#include "kitsune/Transforms/PreLowerAnnotate.h"
#include "kitsune/Transforms/PrefetchForDevice.h"
#include "kitsune/Transforms/PrepareTapirLoops.h"
#include "kitsune/Transforms/RecomputeKernelProperties.h"
#include "kitsune/Transforms/SecondaryIVElimination.h"
#include "kitsune/Transforms/Serialize.h"

#endif // KITSUNE_PASSES_PASSES_H
