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

def ErrAttrAdd
  : Error<"cannot add attribute '{}' without values">;

def ErrAttrNotCompatible
  : Error<"attribute '{}': not compatible with '{}'">;

def ErrAttrBadValues
  : Error<"attribute '{}': incorrect value count '{}' (expecting '{}')">;

def ErrAttrBadValue
  : Error<"attribute '{}': invalid value. {}">;

def ErrAttrBadValueAt
  : Error<"attribute '{}': invalid value at index '{}'. {}">;

def ErrAttrNoValue
  : Error<"attribute '{}': missing value of type '{}'">;

def ErrAttrNoValueAt
  : Error<"attribute '{}': missing value of type '{}' at index '{}'">;

def ErrAttrGlobalNoName
  : Error<"global with attribute '{}': missing required name">;

def ErrAttrGlobalBadType
  : Error<"global with attribute '{}': incorrect type">;

def ErrAttrGlobalNoInit
  : Error<"global with attribute '{}': missing required initializer">;

def ErrAttrGlobalBadInit
  : Error<"global with attribute '{}': invalid initializer. {}">;

def ErrTooManyBitCodeGlobals
  : Error<"too many embedded bitcode globals for tapir target '{}'">;

def ErrTooManyDeviceCodeGlobals
  : Error<"too many embedded device code globals for tapir target '{}'">;

def ErrMissingDeviceCodeGlobal
  : Error<"missing device code global for tapir target '{}'">;

def ErrEmbModuleGeneric
  : Error<"embedded module: {}">;

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

def ErrTapirLoopHeaderTerminator
  : Error<"tapir loop header terminator must be a detach instruction">;

def ErrTapirLoopHeaderInst
  : Error<"tapir loop header must contain only phi nodes">;

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

def ErrTapirLoopNoCanonicalIV
  : Error<"tapir loop does not have a canonical induction variable">;

def ErrTapirNestBoundsVariantGPU
  : Error<"bounds of GPU loop must be invariant in the nest">;

def ErrLoopNotSimplifyForm
  : Error<"loop is not in loop-simplify form">;

def ErrLoopNotLCSSAForm
  : Error<"loop is not in LCSSA form">;

def ErrLoopNotInRotateForm
  : Error<"loop is not in loop-rotate form">;

def ErrTapirLoopNoTask
  : Error<"cannot get task for tapir loop">;

def ErrTapirLoopNotSafeToClone
  : Error<"tapir loop is not safe to clone">;

def ErrTapirLoopConvergent
  : Error<"tapir loop contains convergent operation: '{}'">;

def ErrTapirLoopNoPrimaryIV
  : Error<"primary induction variable not found in tapir loop">;

def ErrTapirLoopNoFiniteTripCount
  : Error<"could not compute finite trip count for tapir loop">;

def ErrTapirLoopThrowsException
  : Error<"tapir loop may not throw an exception">;

def ErrTapirLoopIVUsedOutsideLoop
  : Error<"tapir loop induction variable used outside tapir loop">;

def ErrTapirLoopIVUsesInLatch
  : Error<"tapir loop induction variable must have exactly one use in latch">;

def ErrTapirLoopIVNotUpdatedInLatch
  : Error<"tapir loop induction variable not updated in latch">;

// The first argument is the "kind" of a tapir loop block, e.g. 'header' or
// 'latch'. The second argument is the "kind" of the terminator, e.g.
// 'conditional branch', or 'unconditional branch'.
def ErrTapirLoopUnexpectedTerminator
  : Error<"tapir loop {} must be terminated by a {}">;

def ErrReduceUnitValMismatch
  : Error<"type mismatch between value being reduced and unit">;

def ErrVarArgCallRequiredType
  : Error<"invalid argument in call '{}' at '{}'. Expected value of type '{}'">;

def ErrNumCallArgs
  : Error<"incorrect number of arguments to call '{}'. Expected {}. Got {}">;

def ErrMobileInitExpectSize
  : Error<"call '{}' requires an additional size argument of type '{}'">;

def ErrMobileInitExpectPointer
  : Error<"initializer argument in call '{}' must be a pointer">;

def ErrMobileInitExpectExtra
  : Error<"call '{}' requires additional size argument">;

def ErrMobileInitUnsupportedType
  : Error<"unsupported initializer type '{}' in call '{}'">;

def ErrReducerMinArgs
  : Error<"incorrect number of params in reducer '{}'. Expected 2">;

def ErrReducerReturnType
  : Error<"reducer '{}' must return void">;

def ErrReducerTypeMismatch
  : Error<"type mismatch in reducer '{}' at param '{}'. Expected '{}'">;

def ErrReducerNumParams
  : Error<"incorrect number of params in reducer '{}'. Expected {}. Got {}">;

def ErrKitIntrNoTTID
  : Error<"first argument to call is not a valid TTID">;

def ErrKitIntrWrongTTID
  : Error<"TTID argument in call must be '{}'">;

def ErrSecondaryIVType
  : Error<"secondary induction variable has unsupported type '{}'">;

def ErrSecondaryIVNotBinOp
  : Error<"secondary induction variable has unsupported incoming value '{}'">;

def ErrSecondaryIVOperator
  : Error<"secondary induction variable has unsupported operator '{}'">;

def ErrSecondaryIVNonConstStep
  : Error<"secondary induction variable has non-constant step">;

// Generic errors that can be used to emit any error message.
def ErrGeneric
  : Error<"{}">;

def ErrAttrGeneric
  : Error<"attribute '{}': {}">;

def ErrNYI
  : Error<"NOT YET IMPLEMENTED: {}">;


// ------------------------- Kitsune-specific warnings -------------------------
//
// Although not strictly required, the names of all warnings should be prefixed
// with Warn.

def WarnSerialInParallelLoopNest
  : Warning<"serial loop between parallel loops in loop nest">;

def WarnParallelLoopImperfectlyNested
  : Warning<"parallel loop not perfectly nested">;

// Generic warning that can be used to emit any warning message.
def WarnGeneric
  : Warning<"{}">;


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

def NoteFromArgument
  : Note<"from argument '{}' of '{}'">;

def NoteFromAttribute
  : Note<"from attribute '{}' of '{}'">;

def NoteFromBasicBlock
  : Note<"from basic block '{}'">;

def NoteFromFunction
  : Note<"from function '{}'">;

def NoteFromGlobalVariable
  : Note<"from global variable '{}'">;

def NoteFromInstruction
  : Note<"from instruction '{}'">;

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
