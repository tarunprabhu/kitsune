//===- CBuiltinDocs.td - Docs for C-like language builtins -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Documentation for Kitsune-specific builtins for C-like languages.
//
//===----------------------------------------------------------------------===//

// The category to which the intrinsic belongs. This is only used when
// organizing the documentation.
class KitDocCategory {
  // The text to be used in the section header when generating the
  // documentation.
  string Header;

  // The text for the introductory paragraph for the section containing
  // intrinsics in this category.
  code Intro;
}

// Memory allocation builtins.
def KitDocCatMemAlloc : KitDocCategory {
  let Header = "Memory Allocation Builtins";
  let Intro = [{
These builtins specifically allocate and free
:doc:`mobile buffers <MemoryManagement>`.
}];
}

// Unsafe builtins.
def KitDocCatUnsafe : KitDocCategory {
  let Header = "Unsafe Builtins";
  let Intro = [{
These builtins are intended only for use in Kitsune's runtime and should not be
used elsewhere unless you really know what you are doing. Incorrect use of these
will not raise a compile-time error or warning, but may cause compiler crashes
or, at worst, miscompilation.
}];
}

// Experimental builtins
def KitDocCatExperimental : KitDocCategory {
  let Header = "Experimental Builtins";
  let Intro = [{
These builtins may be used as testbeds for what future language constructs,
controls for Kitsune's compiler passes that cannot be tweaked in any other way,
or controls for Kitsune's runtime.

**WARNING**: The signatures for these builtins may change dramatically at short
notice, and they may even be removed at any time.
}];
}

// Documentation for the parameter of a builtin.
class KitDocParam<string NameParam, string DescrParam> {
// The name of the parameter.
  string Name = NameParam;

  // Description of the parameter.
  string Descr = DescrParam;
}

// Documentation for a Kitsune-specific builtin.
class KitDocCBuiltin {
  // The category to which this intrinsic belongs. This is only used when
  // generating the HTML documentation.
  KitDocCategory Category;

  // Documentation for the intrinsic function parameters. This must, obviously,
  // be kept consistent with the builtin's signature.
  list<KitDocParam> ParamDocs;

  // Description of the builtin.
  code Descr;
}

def KitMobileAllocDoc : KitDocCBuiltin {
  let Category = KitDocCatMemAlloc;
  let ParamDocs = [
      KitDocParam<"n", "The number of bytes to allocate">
  ];
  let Descr = [{
Allocate a mobile buffer. If the buffer could not be allocated for any reason,
this will return `nullptr`.

When generating LLVM IR, this will be lowered to a call to Kitsune's
`llvm.kit.mobile.alloc` intrinsic.
}];
}

def KitMobileFreeDoc : KitDocCBuiltin {
  let Category = KitDocCatMemAlloc;
  let ParamDocs = [
      KitDocParam<"ptr", "The mobile buffer to deallocate">
  ];
  let Descr = [{
Deallocate a mobile buffer previously allocated by
:ref:`kitsune_mobile_alloc`. Passing this builtin a pointer that was not
previously allocated by `kitsune_mobile_alloc` may result in a catastrophic
error.
}];
}

def KitMobileCastUnsafeDoc : KitDocCBuiltin {
  let Category = KitDocCatUnsafe;
  let ParamDocs = [
      KitDocParam<"ptr", "The pointer to cast">
  ];
  let Descr = [{
Cast a pointer to be a mobile pointer. Essentially, this is equivalent to the
code below (which is explicitly disallowed in Kitsune):

.. code-block :: c++

  result = (void* [[kitsune::mobile]]) ptr;

This is only intended to be used by Kitsune's runtime.

.. warning::

  The argument `ptr` must be to a mobile buffer and must have been allocated
  using one of Kitsune's mobile buffer allocators. Failure to do so can result
  in compiler crashes, or, at worst, miscompilation.

  DO NOT use this unless you know what you are doing.
}];
}

def KitReduceDoc : KitDocCBuiltin {
  let Category = KitDocCatExperimental;
  let ParamDocs = [
      KitDocParam<"dest", "Destination of the reduction">,
      KitDocParam<"op", "The reduction operator">,
      KitDocParam<"v", "The value to be reduced using `op`">
  ];
  let Descr = [{
Reduce the value `v` into `dest` using the operator `op`. Currently, the
following operators are supported. Note that each operator only works on a
limited subset of types.

======================= ===================== =================
Operator                Description           Supported Types
======================= ===================== =================
:code:`KIT_AND`         Bitwise AND           Boolean, Integers
:code:`KIT_OR`          Bitwise OR            Boolean, Integers
:code:`KIT_XOR`         Bitwise XOR           Boolean, Integers
:code:`KIT_ADD`         Sum                   Floats, Integers
:code:`KIT_MUL`         Product               Floats, Integers
:code:`KIT_MAX`         Maximum of two values Floats, Integers
:code:`KIT_MAXIMUM`
:code:`KIT_MAXIMUM_NUM`
:code:`KIT_MIN`         Minimum of two values Floats, Integers
:code:`KIT_MINIMUM`
:code:`KIT_MINIMUM_NUM`
======================= ===================== =================

Each entry in the 'Supported Types' column above is a set. The table below
lists the types in each of these sets.

======== =======================================================================
Set      Types
======== =======================================================================
Boolean  :code:`bool`
Floats   :code:`float double`
Integers :code:`int8_t uint8_t int16_t uint16_t int32_t uint32_t int64_t uint64_t`
======== =======================================================================

In addition, some operators are declared, but are not yet supported.

================== ===================================
Operator           Description
================== ===================================
:code:`KIT_CUSTOM` Reduce with a custom reducer
================== ===================================

This builtin *must* be used when a reduction occurs in a parallel for loop.
Consider the loop below

.. code-block :: c++

  forall (int i = 0; i < n; ++i) {
      a += i;
  }

Here, a reduction into `a` is being performed in a parallel loop. Clearly, this
is a race. To ensure that this reduction is performed in parallel, the statement
performing the reduction must be replaced with a call to this builtin as shown
below.

.. code-block :: c++

  forall (int i = 0; i < n; ++i) {
      __kitsune_reduce(&a, KIT_ADD, i);
  }

Although the intrinsic is declared vararg, currently exactly one argument must
be provided after `op`. In the future, when custom reducer functions are
supported, additional parameters required by the reducer may be provided in
addition to `v`, the value to be reduced.
}];
}
