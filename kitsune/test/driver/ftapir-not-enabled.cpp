// Check that the correct error is emitted if a valid tapir target is specified
// but said target has not been enabled.
//
// RUN: %if kitsune-no-cuda %{ \
// RUN:   not %kitxx -fsyntax-only --tapir=cuda %s 2>&1 \
// RUN:       | FileCheck %s -check-prefix CUDA \
// RUN: %}
//
// RUN: %if kitsune-no-hip %{ \
// RUN:   not %kitxx -fsyntax-only --tapir=hip %s 2>&1 \
// RUN:       | FileCheck %s -check-prefix HIP \
// RUN: %}
//
// RUN: %if kitsune-no-lambda %{ \
// RUN:   not %kitxx -fsyntax-only --tapir=lambda %s 2>&1 \
// RUN:       | FileCheck %s -check-prefix LAMBDA \
// RUN: %}
//
// RUN: %if kitsune-no-omptask %{ \
// RUN:   not %kitxx -fsyntax-only --tapir=omptask %s 2>&1 \
// RUN:       | FileCheck %s -check-prefix OMPTASK \
// RUN: %}
//
// RUN: %if kitsune-no-opencilk %{ \
// RUN:   not %kitxx -fsyntax-only --tapir=opencilk %s 2>&1 \
// RUN:       | FileCheck %s -check-prefix OPENCILK \
// RUN: %}
//
// RUN: %if kitsune-no-openmp %{ \
// RUN:   not %kitxx -fsyntax-only --tapir=openmp %s 2>&1 \
// RUN:       | FileCheck %s -check-prefix OPENMP \
// RUN: %}
//
// RUN: %if kitsune-no-qthreads %{ \
// RUN:   not %kitxx -fsyntax-only --tapir=qthreads %s 2>&1 \
// RUN:       | FileCheck %s -check-prefix QTHREADS \
// RUN: %}
//
// RUN: %if kitsune-no-realm %{ \
// RUN:   not %kitxx -fsyntax-only --tapir=realm %s 2>&1 \
// RUN:       | FileCheck %s -check-prefix REALM \
// RUN: %}
//
// CUDA: tapir target 'cuda' was not enabled when kitsune was built
// HIP: tapir target 'hip' was not enabled when kitsune was built
// LAMBDA: tapir target 'lambda' was not enabled when kitsune was built
// OMPTASK: tapir target 'omptask' was not enabled when kitsune was built
// OPENCILK: tapir target 'opencilk' was not enabled when kitsune was built
// OPENMP: tapir target 'openmp' was not enabled when kitsune was built
// QTHREADS: tapir target 'qthreads' was not enabled when kitsune was built
// REALM: tapir target 'realm' was not enabled when kitsune was built
