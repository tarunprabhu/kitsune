//===- CAttrs.td - Kitsune-specific attributes -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is expected to be included in clang/Basic/Attr.td and references
// entities that are only defined there. Using this directly is likely to result
// in some error. In almost all cases, clang/Basic/Attr.td should be used.
//
//===----------------------------------------------------------------------===//

def TT : StmtAttr {
  let Spellings = [CXX11<"tapir", "target">];

  // The target attribute is currently set to be valid on all statements because
  // of the need to add this to Kokkos statements. Because of the way Kokkos is
  // implemented, even if an attribute is added to something that looks like a
  // call in the source, in the AST it ends up being attached to an entirely
  // different expression. We have to check that the attribute is attached to a
  // supported statement in the custom semantic check for the attribute.
  //
  // The enum arguments here are in alphabetical order, not in the same order
  // as the corresponding TTID's. There is no reason for them to be in the same
  // order, so keeping it alphabetical is a bit more organized.
  let Subjects = SubjectList<[Stmt], ErrorDiag, "'parallel' statements">;
  let Args = [EnumArgument<"TT", "TTAttrKind", /*is_string=*/ true,
                           ["cuda", "custom", "hip", "nolo", "opencilk",
                            "openmp", "pthreads", "qthreads", "serial"],
                           ["Cuda", "Custom", "Hip", "Nolo", "OpenCilk",
                            "OpenMP", "Pthreads", "Qthreads", "Serial"]>];
  let Documentation = [TTDocs];
}

def KitsuneLaunch : StmtAttr {
  let Spellings = [CXX11<"kitsune", "launch">];

  // For now, we have no plans to support this on Kokkos constructs, so we can
  // restrict it to be pertain only to forall statements. If that changes, we
  // will have to adopt an approach similar to that used for the tapir target
  // attribute.
  let Subjects = SubjectList<[ForallStmt, CXXForallRangeStmt],
                             ErrorDiag, "'forall' statement">;
  let Args = [IntArgument<"ThreadsPerBlock">];
  let Documentation = [KitsuneLaunchDocs];
}

def KitsuneMemAccess : TypeAttr {
  // FIXME: Would it be better to use a namespaced attribute name here instead
  // of a custom keyword?
  let Spellings = [CustomKeyword<"_readonly">,
                   CustomKeyword<"_writeonly">,
                   CustomKeyword<"_readwrite">];

  let Subjects = SubjectList<[GlobalVar, ParmVar],
                             ErrorDiag>;
  let Accessors = [
    Accessor<"isReadOnly", [CustomKeyword<"_readonly">]>,
    Accessor<"isWriteOnly", [CustomKeyword<"_writeonly">]>,
    Accessor<"isReadWrite", [CustomKeyword<"_readwrite">]>
  ];

  let Documentation = [KitsuneMemAccessDocs];
}

def KitsuneMobile : TypeAttr {
  let Spellings = [GNU<"kitsune_mobile">, CXX11<"kitsune", "mobile">,
                   C23<"kitsune", "mobile">];
  let Documentation = [KitsuneMobileDocs];
}

def KitsuneStmtName : StmtAttr {
  let Spellings = [CXX11<"kitsune", "name">, C23<"kitsune", "name">];
  let Subjects = SubjectList<[ForallStmt, CXXForallRangeStmt],
                             ErrorDiag, "'forall' statement">;
  let Args = [StringArgument<"Name">];
  let Documentation = [KitsuneStmtNameDocs];
}
