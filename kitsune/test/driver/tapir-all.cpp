// This should test all the tapir targets that may be provided on the command
// line. It should be updated when a new tapir target is added, but there is
// currently no way to enforce this. We do not check for -ftapir here because
// that option is officially deprecated and may be removed at some point.
//
// -----------------------------------------------------------------------------
// The tests below should return a success code.
//
// RUN: %kitxx -### --tapir=nolo -O1 %s
// RUN: %kitxx -### --tapir=serial -O1 %s
// RUN: %kitxx -### --tapir=pthreads -O1 %s
// RUN: %kitxx -### --tapir=custom --tapir-plugin=plugin-file -O1 %s
// RUN: %if kitsune-cuda %{ \
// RUN:     %kitxx -### --tapir=cuda --tapir-cuda-arch=sm_80 -O1 %s \
// RUN: %}
// RUN: %if kitsune-hip %{ \
// RUN:     %kitxx -### --tapir=hip --tapir-hip-arch=gfx90c -O1 %s \
// RUN: %}
// RUN: %if kitsune-opencilk %{ \
// RUN:     %kitxx -### --tapir=opencilk -O1 %s \
// RUN: %}
// RUN: %if kitsune-qthreads %{ \
// RUN:     %kitxx -### --tapir=qthreads -O1 %s \
// RUN: %}
//
// -----------------------------------------------------------------------------
// Unknown tapir targets provided to --tapir= should return an error.
//
// RUN: not %kitxx -### --tapir=loremipsum -O1 %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=ERROR
//
// RUN: not %kitxx -### --tapir= -O1 %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=ERROR
//
// ERROR: invalid value '{{.*}}' in '--tapir={{.*}}'
//
// -----------------------------------------------------------------------------
// The tapir targets below have implementations and some measure of support in
// the code. But they have not been maintained and may have bit-rotted. They
// are, therefore, disabled with limited support even in the build system. If
// any are ever resurrected, they should be moved to the first set of known
// tapir targets.
//
// RUN: not %kitxx -### --tapir=lambda -O1 %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=NOT-ENABLED
// RUN: not %kitxx -### --tapir=omptask -O1 %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=NOT-ENABLED
// RUN: not %kitxx -### --tapir=realm -O1 %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=NOT-ENABLED
//
// NOT-ENABLED: tapir target '{{.+}}' was not enabled
//
// -----------------------------------------------------------------------------
// Unlike the tapir targets in the list above, these are likely to be removed
// completely and not resurrected. If that happens, they should be removed from
// here.
//
// RUN: not %kitxx -### --tapir=openmp -O1 %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=NOT-ENABLED
