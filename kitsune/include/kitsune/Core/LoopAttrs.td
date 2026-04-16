//===- LoopAttrs.td - Defines loop attributes --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the attributes that may be added to loops.
//
//===----------------------------------------------------------------------===//

include "kitsune/Core/AttrCommon.td"
include "kitsune/Core/LoopAttrDocs.td"

// Base class for all Kitsune-specific loop attributes. These are most likely to
// be added to tapir loops, but they may be added to regular loops as well.
// These will, necessarily, be interspersed with other loop "attributes" added
// by LLVM such as "loop.unroll.*" and others that are not specific to tapir
// loops.
//
// The names of the concrete instances of this class will be the name of the
// attribute as it appears in the LoopAttrKind enum. This must be in Pascal
// case, and consist of letters and digits only. The name of the attribute as it
// appears in LLVM-IR metadata is derived from this.
//
// If AllowedOn is set to TapirOnly, the attributes would appear in metadata as
// shown below.
//
//     .-----------------------------------------.
//     | Name         | Metadata string          |
//     |-----------------------------------------|
//     | Target       | tapir.loop.target        |
//     | PerfectDepth | tapir.loop.perfect.depth |
//     | Grainsize    | tapir.loop.grainsize     |
//     | Dim3         | tapir.loop.dim3          |
//     '-----------------------------------------'
//
// If AllowedOn is set to any value other than TapirOnly, the attributes would
// appear in metadata as shown below.
//
//     .-----------------------------------.
//     | Name         | Metadata string    |
//     |-----------------------------------|
//     | Target       | loop.target        |
//     | PerfectDepth | loop.perfect.depth |
//     | Grainsize    | loop.grainsize     |
//     | Dim3         | loop.dim3          |
//     '-----------------------------------'
//
class LoopAttr<Type ValueType = VoidTy> : AttrBase<ValueType>{}

//------------------------------------------------------------------------------

// The name of a loop - usually a tapir loop. This can be useful for debugging
// or when one is compelled to read LLVM-IR.
def Name : LoopAttr<StrTy> {
  let HasCustomVerifier = true;
  let Documentation = KitLoopNameDoc;
}

// Flag indicating that the loop is a serialized tapir loop.
def Serialized : LoopAttr<> {
   let Documentation = KitLoopSerializedDoc;
}

// Flag indicating that the tapir loop is lowered to a GPU kernel, and Kitsune's
// runtime will perform autotuning on the launch parameters when launching the
// GPU kernel generated from this tapir loop.
// TODO: Rename this to AutotuneKernelLaunch
def AutotuneLaunch : LoopAttr<> {
  let Documentation = KitLoopAutotuneLaunchDoc;
}

// The grainsize to use when lowering the tapir loop.
// We name this Grainsize to ensure that the name of the generated attribute is
// tapir.loop.grainsize. This is backwards compatible with the name of the
// attribute in OpenCilk. We may change this in the future to be more consistent
// with the other attribute names, but that would require changes to most tests
// in llvm/lib/Transforms/Tapir.
def Grainsize : LoopAttr<U32Ty> {
  let Documentation = KitLoopGrainsizeDoc;
}

// Flag indicating that the loop spawning pass should lowrer the tapir loop
// using the tapir target specified in the loop's target attribute. Tapir loops
// that do not have this attribute set will not be lowered by the loop-spawning
// pass.
def LoweringEnabled : LoopAttr<> {
  let Documentation = KitLoopLoweringEnabledDoc;
}

// The number of perfectly nested tapir loops rooted at the given loop.
def PerfectDepth : LoopAttr<U32Ty> {
  let Documentation = KitLoopPerfectDepthDoc;
}

// The nesting level of a loop that is perfectly nested within a loop nest.
def PerfectLevel : LoopAttr<U32Ty> {
  let Documentation = KitLoopPerfectLevelDoc;
}

// The spawn strategy for the tapir loop.
def SpawnStrategy : LoopAttr<SpawnStrategyTy> {
  let Documentation = KitLoopSpawnStrategyDoc;
}

// The tapir target to use when lowering the tapir loop.
def Target : LoopAttr<TTIDTy> {
  let Documentation = KitLoopTargetDoc;
}

// The number of threads per block to use when launching a GPU kernel generated
// from a tapir loop nest.
def ThreadsPerBlock : LoopAttr<U32Ty> {
  let HasCustomVerifier = true;
  let Documentation = KitLoopThreadsPerBlockDoc;
}
