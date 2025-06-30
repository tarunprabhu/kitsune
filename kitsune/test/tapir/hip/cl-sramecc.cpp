// Check that the --tapir-hip-sramecc option is handled correctly.
//
// -----------------------------------------------------------------------------
// RUN: %kitxx -### --tapir=hip --tapir-hip-sramecc=on -O1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,ON
//
// RUN: %kitxx -### --tapir=hip --tapir-hip-sramecc=off -O1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,OFF
//
// RUN: %kitxx -### --tapir=hip --tapir-hip-sramecc=any -O1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,ANY
//
// ALL: -cc1
// ON: --tapir-hip-sramecc=on
// OFF: --tapir-hip-sramecc=off
// ANY: --tapir-hip-sramecc=any
//
// -----------------------------------------------------------------------------
//
// RUN: not %kitxx -### --tapir=hip --tapir-hip-sramecc= -O1 %s 2>&1  \
// RUN:     | FileCheck %s -check-prefix MISSING
//
// MISSING: error: argument to '--tapir-hip-sramecc=' is missing
//
// -----------------------------------------------------------------------------
//
// RUN: not %kitxx -### --tapir=hip --tapir-hip-sramecc=ignore -O1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix IGNORE
//
// IGNORE: error: invalid argument 'ignore' to -tapir-hip-sramecc=
//
// -----------------------------------------------------------------------------
