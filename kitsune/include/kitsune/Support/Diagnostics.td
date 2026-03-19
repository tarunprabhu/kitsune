//===- Diagnostics.td - Kitsune-specific diagnostics -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the diagnostics that Kitsune may emit from the middle-end
//
//===----------------------------------------------------------------------===//

// Base class indicating the severity of a diagnostic. Each instance of this
// class represents a different severity.
class Severity;

// A note. This is intended to add additional information to a previously
// emitted diagnostic - usually a remark.
def SNote : Severity;

// A remark. This is used to indicate the result of an analysis or
// transformation performed by Kitsune.
def SRemark : Severity;

// A warning.
def SWarn : Severity;

// An error.
def SErr : Severity;

// A diagnostic message.
class Diag<Severity SevParam, string MsgParam> {
  // The severity of the diagnostic.
  Severity Sev = SevParam;

  // The message that this diagnostic emits. This may be a format string that
  // follows the grammar supported by llvm::formatv.
  string Msg = MsgParam;
}

// Base class for all errors.
class Error<string Msg> : Diag<SErr, Msg>;

// Base class for all warnings.
class Warning<string Msg> : Diag<SWarn, Msg>;

// Base class for all remarks.
class Remark<string Msg> : Diag<SRemark, Msg>;

// Base class for all notes.
class Note<string Msg> : Diag<SNote, Msg>;


// -------------------------- Kitsune-specific errors --------------------------
//
// Although not strictly required, the names of all errors should be prefixed
// with Err.

def ErrAttrWithoutValues
  : Error<"Cannot add attribute '{}' without values">;

def ErrRequiredPassNotRun
  : Error<"'{}': required pass '{}' has not been run">;

def ErrParseEmbBC
  : Error<"could not parse embedded bitcode in global '{}': {}">;

def ErrParseLLVM
  : Error<"could not parse LLVM file: {}">;

def ErrLinkLLVM
  : Error<"could not link LLVM file: {}">;

def ErrOptOccurrences
  : Error<"option '--{}' must be provided exactly once">;

def ErrOptValueMissing
  : Error<"option '--{}' requires at least one valid value">;

def ErrOptValueInvalid
  : Error<"option '--{}' has invalid value '{}'">;

def ErrOptValueInvalidRange
  : Error<"option '--{}' has value '{}' not in range [{},{}]">;

def ErrTTPluginAPIVersion
  : Error<"wrong API version on plugin '{}'. Got version {}, supported version is {}">;

def ErrTTPluginCBConstructor
  : Error<"missing constructor callback in plugin '{}'">;

def ErrTTPluginCBCompilerOpts
  : Error<"missing compiler options callback in plugin '{}'">;

def ErrTTPluginCBLinkerOpts
  : Error<"missing linker options callback in plugin '{}'">;

def ErrTTPluginEntryPoint
  : Error<"plugin entry point not found in '{}'">;

def ErrTTPluginLoad
  : Error<"could not load plugin library '{}': {}">;

def ErrTTNotEnabled
  : Error<"required tapir target '{}' has not been enabled">;

def ErrTTMultiple
  : Error<"compiling with multiple tapir targets is not yet supported">;

def ErrTTIncompatible
  : Error<"tapir targets '{}' and '{}' are incompatible">;

def ErrTTIncompatibleLoop
  : Error<"tapir targets '{}' and '{}' are incompatible in loop nest">;

def ErrTTIncompatibleLoopGPU
  : Error<"tapir targets '{}' and '{}' are incompatible in GPU loop nest">;

def ErrTapirInstNotInTapirLoop
  : Error<"tapir instructions outside tapir loops are not yet supported">;

def ErrTapirLoopSingleIndVar
  : Error<"tapir loop must have at most one induction variable">;

def ErrTapirLoopNoUniqueDetachInst
  : Error<"tapir loop does not have a unique detach instruction">;

def ErrTapirLoopNoUniqueReattachInst
  : Error<"tapir loop does not have a unique reattach instruction">;

def ErrTapirLoopNoUniqueSyncInst
  : Error<"tapir loop does not have a unique sync instruction">;

def ErrTapirLoopNoUniqueSyncRegion
  : Error<"tapir loop not associated with a unique sync region">;

def ErrTapirLoopSyncRegionDefn
  : Error<"syncregion is not the result of an intrinsic call">;

def ErrTapirLoopSyncRegionUses
  : Error<"unexpected use of sync region associated with tapir loop">;

def ErrTapirLoopSyncMustPostDominate
  : Error<"tapir loop not post-dominated by sync instruction">;

def ErrTapirLoopNotCanonicalGPU
  : Error<"tapir loop for GPU is not canonical">;

def ErrTapirNestBoundsVariantGPU
  : Error<"bounds of GPU loop must be invariant in the nest">;


// ------------------------- Kitsune-specific warnings -------------------------
//
// Although not strictly required, the names of all warnings should be prefixed
// with Warn.

def WarnSerialInParallelLoopNest
  : Warning<"serial loop between parallel loops in loop nest">;

def WarnParallelLoopImperfectlyNested
  : Warning<"parallel loop not perfectly nested">;


// ------------------------- Kitsune-specific remarks -------------------------
//
// Although not strictly required, the names of all remarks should be prefixed
// with Remark.

def RemarkSerializedLoop
  : Remark<"serialized loop">;


// -------------------------- Kitsune-specific notes --------------------------
//
// Although not strictly required, the names of all notes should be prefixed
// with Note.

def NoteFromBasicBlock
  : Note<"from basic block '{}'">;

def NoteFromFunction
  : Note<"from function '{}'">;

def NoteFromLoop
  : Note<"from loop '{}'">;

def NoteLoopNestRoot
  : Note<"root of tapir loop nest is '{}'">;

def NoteAncestorLoopTarget
  : Note<"target on ancestor loop is '{}'">;

def NoteDescendantLoopTarget
  : Note<"target on descendant loop is '{}'">;

def NoteSerializedLoop
  : Note<"serialized loop: {}">;
