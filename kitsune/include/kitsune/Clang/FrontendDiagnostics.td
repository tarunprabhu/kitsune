//===- CFrontendDiagnostics.td - Kitsune C/C++ diagnostics -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Diagnostics emitted by Kitsune's C/C++ frontends, typically during "codegen",
// i.e. LLVM-IR generation.
//
// This file is intended to be included in
// clang/Basic/DiagnosticFrontendKinds.td. It references entities defined in
// that file, or those included transitively from it. This should not,
// generally, be used directly. DiagnosticFrontendKinds.td should be used
// instead.
//
//===----------------------------------------------------------------------===//

def warn_kokkos_range_expr : Warning<
    "kokkos: unsupported expression for range of parallel_for. "
    "Using standard C++">,
    InGroup<Kitsune>;

def warn_kokkos_no_functor : Warning<
    "kokkos: functors not supported in parallel_for. Using standard C++">,
    InGroup<Kitsune>;

def warn_kokkos_no_lambda_args : Warning<
    "kokkos: lambda in parallel_for requires arguments. Using standard C++">,
    InGroup<Kitsune>;

def warn_kokkos_unknown_stmt_class : Warning<
    "kokkos: unrecognized/unsupported construct in parallel_for statement. "
    "Using standard C++">,
    InGroup<Kitsune>;

def warn_kokkos_unknown_bounds_expr : Warning<
    "kokkos: unrecognized bounds/limit expression. Using standard C++">,
    InGroup<Kitsune>;

def warn_kokkos_reduce_unsupported : Warning<
    "kokkos: parallel_reduce is not supported. Using standard C++">,
    InGroup<Kitsune>;
