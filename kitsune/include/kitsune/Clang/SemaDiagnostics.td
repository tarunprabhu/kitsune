//===- CSemaDiagnostics.td - Kitsune's sema diagnostics ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Diagnostics emitted by Kitsune's C/C++ frontends during semantic analysis.
//
// This file is intended to be included in clang/Basic/DiagnosticSemaKinds.td.
// It references entities defined in that file, or those included transitively
// from it. This should not, generally, be used directly. DiagnosticSemaKinds.td
// should be used instead.
//
//===----------------------------------------------------------------------===//

// General errors that may apply to all Kitsune-specific attributes
def err_duplicate_attribute_stmt : Error<
    "attribute %0 cannot appear more than once on a statement">;

// tapir::target attribute
def err_tapir_target_attr_unsupported_stmt: Error<
    "'tapir::target' attribute: unsupported statement">;
def err_tapir_target_unknown: Error<
    "'tapir::target' attribute: unknown value ">;
def err_tapir_target_custom: Error<
    "'tapir::target' attribute: value cannot be 'custom'">;

// kernel::launch attribute
def err_kitsune_launch_range: Error<
    "'kitsune::launch' attribute: value not in range [0,1024]">;

// spawn statement
def warn_spawn_as_loop_body: Warning<
    "%0 loop with spawn statement body has undefined behavior">,
    InGroup<Kitsune>;
def err_spawn_in_forall: Error<
    "spawn statements are not allowed in forall loops">;

// forall statement
def err_forall_no_init: Error<
    "forall statement must have an initialization expression">;
def err_forall_no_cond: Error<
    "forall statement must have a condition expression">;
def err_forall_no_inc: Error<
    "forall statement must have an increment expression">;
def err_forall_has_break: Error<
    "forall body may not have a break statement">;
def err_forall_init_not_decl: Error<
    "initializer in a forall statement must be a variable declaration">;
def err_forall_init_multiple: Error<
    "initializer in a forall statement must declare exactly one variable">;

// memory access qualifiers
def err_kitsune_multiple_access_qualifiers : Error<
    "multiple access qualifiers">;
def note_kitsune_typedef_access_qualifier : Note<
    "previously declared '%0' here">;

// mobile attribute/qualifier
def err_kitsune_mobile_on_non_pointer : Error<
    "'kitsune::mobile' can only be used on a pointer type">;
def err_kitsune_cast_to_mobile : Error<
    "cast to a kitsune::mobile pointer is not allowed">;
def err_kitsune_cast_away_mobile : Error<
    "%0 cannot be used to cast away mobile qualifier. Use c-style cast "
    "instead">;

def note_ovl_candidate_no_mobile: Note<
    "candidate %sub{select_ovl_candidate_kind}0,1,2 not viable: "
    "%ordinal4 argument (%3) does not accept mobile pointer. Consider using "
    "__kitsune_mobile_cast_unsafe">;
def note_ovl_candidate_with_mobile: Note<
    "candidate %sub{select_ovl_candidate_kind}0,1,2 not viable: "
    "%ordinal4 argument (%3) requires mobile pointer">;

// mobile builtins
def err_kitsune_mobile_free_arg : Error<
    "argument of kitsune_mobile_free must have mobile qualifier">;
def err_kitsune_mobile_cast_arg: Error<
    "argument of __kitsune_mobile_cast_arg must be a non-mobile pointer">;

// reduce builtins
def err_kit_reduce_num_args : Error<
    "incorrect number of arguments in call to __kitsune_reduce. Expect 3">;
def err_kit_reduce_type_unsupported : Error<
    "value being reduced must have builtin scalar type">;
def err_kit_reduce_type_mismatch : Error<
    "reduction destination must be pointer to %0">;
def err_kit_reduce_op_not_literal : Error<
    "reduction operator must be an integer literal">;
def err_kit_reduce_op_type : Error<
    "reduction operator '%0' not valid for values of type %1">;
def err_kit_reduce_op_unknown : Error<
    "unknown reduction operator">;
def err_kit_reduce_op_nyi : Error<
    "reduction operator '%0' is not yet supported">;
