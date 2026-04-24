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
//===----------------------------------------------------------------------===//

// NOTE: This file is expected to be included in clang/Basic/Builtins.td.
// It cannot be used on its own. When passing it to tablegen, for instance, the
// including file must be used instead.

include "kitsune/Frontend/CBuiltinDocs.td"

class KitCBuiltin : Builtin {
  KitDocCBuiltin Doc;
}

// Allocate a mobile buffer.
def KitMobileAlloc : KitCBuiltin {
  let Spellings = ["kitsune_mobile_alloc"];
  let Attributes = [CustomTypeChecking];
  let Prototype = "void!(size_t)"; // Here, `!` means `*[[kitsune::mobile]]`
  let Doc = KitMobileAllocDoc;
}

// Free a mobile buffer that was previously allocated with kitsune_mobile_alloc.
def KitMobileFree : KitCBuiltin {
  let Spellings = ["kitsune_mobile_free"];
  let Attributes = [CustomTypeChecking];
  let Prototype = "void(void!)"; // Here, `!` means `*[[kitsune::mobile]]`
  let Doc = KitMobileFreeDoc;
}

// Effectively add the [[kitsune::mobile]] attribute to a pointer.
def KitMobileCastUnsafe : KitCBuiltin {
  let Spellings = ["__kitsune_mobile_cast_unsafe"];
  let Attributes = [CustomTypeChecking];
  let Prototype = "void!(void*)"; // Here, `!` means `[[kitsune::mobile]]`
  let Doc = KitMobileCastUnsafeDoc;
}
