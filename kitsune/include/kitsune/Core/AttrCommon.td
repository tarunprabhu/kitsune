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

// Documentation for a Kitsune-specific attribute.
class KitDocAttr {
  code Documentation;
}

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
// objects and as LLVM Value's - including metadata. Integer types with explicit
// bitwidths are strongly preferred over `int`, `long` etc. Floating point types
// are also allowed, but here, we use `float` and `double` since explicit
// width types are not provided in C++.
//
// Pointer types are *not* allowed here.
//
class BasicType<string NameParam, bit IsEnumParam = false> : Type {
  string Name = NameParam;
  bit IsEnum = IsEnumParam;
}

// A void type. This should only be used for "flag" attributes i.e. those that
// do not take an actual value.
def VoidTy : BasicType<"void">;

// Builtin integer types.
def U8Ty : BasicType<"uint8_t">;
def U16Ty : BasicType<"uint16_t">;
def U32Ty : BasicType<"uint32_t">;
def U64Ty : BasicType<"uint64_t">;

// Floating point types.
def F32Ty : BasicType<"float">;
def F64Ty : BasicType<"double">;

// A special case for attributes whose value is the ID of a loop. The loop ID is
// a raw MDNode. A pointer to this will be returned by the getter of an
// attribute whose value is of this type.
def LoopIDTy : BasicType<"llvm::MDNode*">;

// We use LLVM's StringRef for all string types.
def StrTy : BasicType<"llvm::StringRef">;

// Enumerations that are used in attributes. In general, we include all enums
// here, but others may be defined in the files with specific attributes.
def TTIDTy : BasicType<"llvm::TTID", true>;
def SpawnStrategyTy : BasicType<"llvm::TapirSpawnStrategy", true>;

//------------------------------------------------------------------------------
//
// A list type. This allows the attribute to have multiple values of the same
// type. Each value of the attribute must be a basic type.
class ListType<BasicType TypeParam> : Type {
  BasicType ElemType = TypeParam;
}

//------------------------------------------------------------------------------
//
// A set type. This allows the attribute to have multiple values of the same
// type, but each value can appear at most once. Each value of the attribute
// must be a basic type.
class SetType<BasicType TypeParam> : Type {
  BasicType ElemType = TypeParam;
}

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

  // If true, the attribute has a custom verifier. In this case, a custom
  // verifier function with the appropriate signature must be provided. The
  // signature of this function depends on the type of the attribute. In the
  // examples below, IRElem is the type of the LLVM IR entity to which the
  // attribute can be attached, e.g. llvm::Function, llvm::GlobalVariable,
  // llvm::Instruction, llvm::Loop, or llvm::Module.
  //
  // For flag attributes i.e. those that do not have a value, this must be as
  // shown below. The hasAttr argument will always be true.
  //
  //     bool verify##NAME##Attr(const IRElem &ir, const bool &hasAttr,
  //                             raw_ostream *os);
  //
  // For attributes that take a single value, this must be as shown below.
  // TYPE is the type of the attribute value. `v` is the value of the attribute.
  //
  //     bool verify##NAME##Attr(const IRElem &ir, const TYPE &ty,
  //                             raw_ostream *os);
  //
  // Attributes that take a Loop value, however, are different. Loop objects can
  // only be obtained by running the loop analysis pass. Since the verification
  // of attributes may be run outside of a pass manager, we cannot always
  // provide a loop object to a custom verifier. Since the "loop" values in such
  // attributes is simply the metadata node corresponding to the loop, that will
  // be provided to the custom verifier.
  //
  //     bool verify##NAME##Attr(const IRElem &ir, const MDNode &loopID,
  //                             raw_ostream *os)
  //
  // For attributes that take multiple values, this must be as shown below.
  // Here, ETY<I> is the type of the I'th value required by the attribute.
  // N is the number of values that the attribute requires.
  //
  //     bool verify##NAME##Attr(const IRElem &ir, const ETY0 &e0, ...
  //                             const ETY<N> &en, raw_ostream *os);
  //
  // In each case, an optional raw_ostream may be provided. In this case, an
  // error message may be printed to it if the attribute is not valid. The
  // verifier must return true if the attribute is valid, false otherwise.
  //
  bit HasCustomVerifier = false;

  // Documentation for this attribute. This is used to generate the HTML docs.
  KitDocAttr Documentation;
}

//------------------------------------------------------------------------------
