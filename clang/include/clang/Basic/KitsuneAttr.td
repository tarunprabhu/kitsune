//===- KitsuneAttr.td - Kitsune-specific attribute definitions ------------===//
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

def TapirTarget : StmtAttr {
  // FIXME: Change the spelling of this attribute to use hte kitsune "namespace"
  let Spellings = [CXX11<"tapir","target">];

  // FIXME KITSUNE: The target attribute is currently set to be valid on all
  // statements because of the need to add this to Kokkos. We should either:
  //
  //   - Remove the Forall statement from the subject list and have it apply to
  //     all statements and check when parsing that it is on a statement that
  //     we actually care about.
  //
  //   - Or, and we should prefer this if possible, identify exactly what the
  //     Kokkos construct is and add just that statement class to this subject
  //     list.
  //
  let Subjects = SubjectList<[ForallStmt, CXXForallRangeStmt, Stmt],
                             ErrorDiag, "'parallel' statements">;
  let Args = [
    EnumArgument<"TapirTargetAttrType", "TapirTargetAttrTy",
                 /*is_string=*/ true,
                 ["nolo", "serial", "cuda", "custom", "hip", "opencilk",
                  "openmp", "pthreads", "qthreads"],
                 ["Nolo", "Serial", "Cuda", "Custom", "Hip", "OpenCilk",
                  "OpenMP", "Pthreads", "Qthreads"],
                 0>
  ];
  let Documentation = [TapirTargetDocs];
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

def KitsuneLaunch : StmtAttr {
  let Spellings = [CXX11<"kitsune", "launch">];
  let Subjects = SubjectList<[ForallStmt, CXXForallRangeStmt],
                             ErrorDiag, "'forall' statement">;
  let Args = [IntArgument<"ThreadsPerBlock">];
  let Documentation = [KitsuneLaunchDocs];
}

def KitsuneMobile : TypeAttr {
  let Spellings = [GNU<"kitsune_mobile">, CXX11<"kitsune", "mobile">,
                   C23<"kitsune", "mobile">];
  let Documentation = [KitsuneMobileDocs];
}
