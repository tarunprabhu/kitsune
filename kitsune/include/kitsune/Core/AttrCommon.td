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

// Basic class for all types for an attribute value. This is intended to be
// "abstract" in the sense that it is not intended to be used directly by
// users i.e. those writing .td files. Instead one of the classes that inherits
// from this should be used.
class Type;

//-----------------------------------------------------------------------------
//
// Base class for the type a single-valued i.e. non-composite attribute. The
// Name field must be spelled exactly as the corresponding C++ type. The type
// must be a builtin type, or an enum defined in the llvm namespace.
// llvm::StringRef's are treated as special cases and are deemed to be "basic"
// since there is excellent support for handling them in as both standard C++
// objects and as LLVM value's - including metadata. Integer types with explicit
// bitwidths are strongly preferred over `int`, `long` etc. Floating point types
// are also allowed, but here, we use `float` and `double` since explicit
// width types are not provided in C++.
//
// Pointer types are *not* allowed here.
//
class BasicType<string NameParam, Bool IsEnumParam = False> : Type {
  string Name = NameParam;
  Bool IsEnum = IsEnumParam;
}

// A void type. This should only be used for "flag" attributes i.e. those that
// do not take an actual value.
def VoidTy : BasicType<"void">;

// Builtin integer types.
def I8Ty : BasicType<"int8_t">;
def I16Ty : BasicType<"int16_t">;
def I32Ty : BasicType<"int32_t">;
def I64Ty : BasicType<"int64_t">;

// Floating point types.
def F32Ty : BasicType<"float">;
def F64Ty : BasicType<"double">;

// We use LLVM's StringRef for all string types.
def StrTy : BasicType<"llvm::StringRef">;

// Enumerations that are used in attributes. In general, we include all enums
// here, but others may be defined in the files with specific attributes.
def TTIDTy : BasicType<"llvm::TTID", True>;
def SpawnStrategyTy : BasicType<"llvm::TapirSpawnStrategy", True>;


//-----------------------------------------------------------------------------
//
// A special case for attributes whose value is a loop. This requires special
// handling, so we create a separate type for it. This cannot be used as an
// element of a tuple type below. This, again is intentional.
def LoopTy : Type {}

//-----------------------------------------------------------------------------
//
// An element of a the tuple type defined below. This consists of a basic type
// and a name. The name is used to retrieve it from the containing tuple.
// The name must be a a valid C identifier.
class TupleElement<BasicType TypeParam, string NameParam> {
  BasicType ElemType = TypeParam;
  string ElemName = NameParam;
}

// A tuple type that consists of an ordered list of basic types. Each element
// of the tuple can be accessed by name. In principle, it can also be accessed
// by index, but we are not likely to support that since it is not clear that
// there is value in doing so.
class TupleType<list<TupleElement> ElementsParam> : Type {
  list<TupleElement> Elements = ElementsParam;
}

//------------------------------------------------------------------------------

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
