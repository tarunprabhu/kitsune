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

.. code::c++

  result = (void* [[kitsune::mobile]]) ptr;

This is only intended to be used by Kitsune's runtime.

.. warning::

  The argument `ptr` must be to a mobile buffer and must have been allocated
  using one of Kitsune's mobile buffer allocators. Failure to do so can result
  in compiler crashes, or, at worst, miscompilation.

  DO NOT use this unless you know what you are doing.
}];
}
