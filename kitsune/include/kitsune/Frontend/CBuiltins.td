//===- CBuiltins.td - Builtins for C-like languages --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Kitsune-specific builtins for C-like languages.
//
// This file is expected to be included in clang/Basic/Builtins.td. To process
// the contents of this file, for instance with tablgen, the including file must
// be passed to tablgen. This *MUST NOT* be used directly.
//
//===----------------------------------------------------------------------===//

// NOTE: In the intrinsic definitions used below, when a mobile pointer must be
// specified, use '*__mobile__'. Note that there is no space between '*' and
// 'mobile'. '*__mobile__' is equivalent to `* [[kitsune::mobile]]`.

include "kitsune/Frontend/CBuiltinDocs.td"

class KitCBuiltin : Builtin {
  KitDocCBuiltin Doc;
}

// Allocate a mobile buffer.
def KitMobileAlloc : KitCBuiltin {
  let Spellings = ["kitsune_mobile_alloc"];
  let Attributes = [];
  let Prototype = "void*__mobile__(size_t)";
  let Doc = KitMobileAllocDoc;
}

// Free a mobile buffer that was previously allocated with kitsune_mobile_alloc.
def KitMobileFree : KitCBuiltin {
  let Spellings = ["kitsune_mobile_free"];
  let Attributes = [CustomTypeChecking];
  let Prototype = "void(void*__mobile__)";
  let Doc = KitMobileFreeDoc;
}

// Effectively add the [[kitsune::mobile]] attribute to a pointer.
def KitMobileCastUnsafe : KitCBuiltin {
  let Spellings = ["__kitsune_mobile_cast_unsafe"];
  let Attributes = [CustomTypeChecking];
  let Prototype = "void*__mobile__(void*)";
  let Doc = KitMobileCastUnsafeDoc;
}
