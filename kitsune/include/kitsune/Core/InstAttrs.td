//===- InstAttrs.td - Defines instruction attributes -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Kitsune-specific attributes that can be added to instructions.
//
//===----------------------------------------------------------------------===//

// Base class for all Kitsune-specific instruction attributes. For now, the
// attributes are not constrained to be applied to only certain instruction
// kinds, but that might change in the future.
//
// The names of the concrete instances of this class will be the name of the
// attribute as it appears in the InstAttrKind enum. This must be in Pascal
// case, and consist of letters and digits only. The name of the attribute as it
// appears in LLVM-IR metadata is derived from this.
//
//     .-----------------------------------------.
//     | Name         | Metadata string          |
//     |-----------------------------------------|
//     | Safe         | kit.safe                 |
//     | ImplicitSync | kit.tapir.loop           |
//     | Reduce3      | kit.reduce3              |
//     '-----------------------------------------'
//
class Attr<string ValueTypeParam> {
  // If the attribute takes a value, this must be the exact spelling of the
  // C++ type of the argument. If the attribute does not take an argument, this
  // must be an empty string.
  string ValueType = ValueTypeParam;

  // Documentation for this attribute. This is used to generate the HTML
  // documentation.
  code Documentation = [{}];
}

// Attribute that takes an enum argument.
class EnumAttr<string ValueType> : Attr<ValueType>;

// Attribute that does not take an argument.
class FlagAttr : Attr<"">;

// Attribute that takes a 32-bit integer argument.
class Int32Attr : Attr<"int32_t">;

// Attribute that takes a 64-bit integer argument.
class Int64Attr : Attr<"int64_t">;

// Attribute that takes a metadata node that is the ID node of a loop as an
// argument.
class LoopAttr : Attr<"Loop*">;

// Attribute that takes a raw metadata node as an argument.
class MDNodeAttr : Attr<"MDNode*">;

// Attribute that takes a string argument.
class StrAttr : Attr<"StringRef">;

//------------------------------------------------------------------------------

def SourceLoop : LoopAttr {
  let Documentation = [{
This attribute should only be attached to instructions that were originally,
i.e. at the time of IR-generation, part of a loop. The value is the metadata
node that is the ID of the loop to which the instruction originally belonged.
This loop will be the "nearest" parent of the instruction. Consider the exampe
below:

.. code-block::c++

  for (i ...) {
      for (j ...) {
          func();
          ...
      }
  }

Here, the value of the SourceLoop metadata attached to the call instruction
(where the callee is `func`) will be the ID of the `j`-loop since it is the
nearest parent loop.

This is expressly intended to be used for the "provenance" of instructions. For
example, if the LICM pass were to hoist the call to `func` from the innermost
loop as shown below:

.. code-block::c++

  for (i ...) {
      func();
      for (j ...) {
          ...
      }
  }

the value of this metadata node is expected to be **unchanged**. With this
transformation, the nearest parent of the call instruction would be the `i`-loop
but the SourceLoop attribute value should still be the `j`-loop.

Note that if the original source loop were to be deleted, the value of this
attribute would be `null`.
}];
}
