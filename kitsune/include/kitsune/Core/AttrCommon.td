//===- AttrCommon.td - Common classes for attributes -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common base classes and types for Kitsune's attributes.
//
//===----------------------------------------------------------------------===//

// Base class whose specializations specify the type of construct to which an
// attribute can be added. Only three specializations of this class are allowed.
// These are listed below
//
//    TapirOnly      Can only be applied to tapir-specific constructs
//    NormalOnly     Can only be applied to regular LLVM constructs
//    TapirOrNormal  Can be applied to both tapir-specific and regular LLVM
//                   constructs
//
class AllowedOnKind;
def TapirOnly : AllowedOnKind;
def NormalOnly : AllowedOnKind;
def TapirOrNormal : AllowedOnKind;

// Boolean type and values since these are not available in TableGen.
class Bool;
def True : Bool;
def False : Bool;

// Base class for the type of an attribute value. The Name field must be spelled
// exactly as the corresponding type. The type must be a builtin type, or one
// defined in the llvm namespace. For instance, int32_t, StringRef or TTID
// (defined in kitsune/Core/Tapir.h). Types with explicit bitwidths are strongly
// preferred over `int`, `long` etc.
//
// If the attribute expects a pointer as an operand, the name must end with '*'.
// There should not be too many cases of this - in most cases, the values should
// be scalars.
//
// Non-scalar, non-pointer types are not supported for values.
class Type<string NameParam, Bool IsEnumParam = False> {
  string Name = NameParam;
  Bool IsEnum = IsEnumParam;
}

// A void type. This should only be used for "flag" attributes i.e. those that
// do not take an actual value.
def VoidTy : Type<"void">;

// Builtin integer types. We intentionally do not support all possible integer
// bit widths.
def I32Ty : Type<"int32_t">;
def I64Ty : Type<"int64_t">;

// Floating point types.
def F32Ty : Type<"float">;
def F64Ty : Type<"double">;

// We use LLVM's StringRef for all string types.
def StrTy : Type<"llvm::StringRef">;

// Enumerations that are used in attributes. In general, we include all enums
// here, but others may be defined in the files with specific attributes.
def TTIDTy : Type<"llvm::TTID", True>;
def SpawnStrategyTy : Type<"llvm::TapirSpawnStrategy", True>;

// -----------------------------------------------------------------------------

// Base class for Kitsune-specific attributes. This can be used for all
// attributes that have at most one value. If the attribute does not have a
// value, the value type must be "void" (use the VoidTy defined above).
class AttrBase<Type ValueTypeParam = VoidTy> {
  // The type of the attribute value. If the attribute does not take a value,
  // this must be VoidTy.
  Type ValueType = ValueTypeParam;

  // Documentation for this attribute. This is used to generate the HTML
  // documentation.
  code Documentation = [{}];
}
