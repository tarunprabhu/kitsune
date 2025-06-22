// Check that the --tapir-hip-xnack option is handled correctly.
//
// -----------------------------------------------------------------------------
// RUN: %kitxx -### --tapir=hip --tapir-hip-xnack=on %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,ON
//
// RUN: %kitxx -### --tapir=hip --tapir-hip-xnack=off %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,OFF
//
// RUN: %kitxx -### --tapir=hip --tapir-hip-xnack=any %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,ANY
//
// ALL: -cc1
// ON: --tapir-hip-xnack=on
// OFF: --tapir-hip-xnack=off
// ANY: --tapir-hip-xnack=any
//
// -----------------------------------------------------------------------------
//
// RUN: not %kitxx -### --tapir=hip --tapir-hip-xnack= %s 2>&1  \
// RUN:     | FileCheck %s -check-prefix MISSING
//
// MISSING: error: argument to '--tapir-hip-xnack=' is missing
//
// -----------------------------------------------------------------------------
//
// RUN: not %kitxx -### --tapir=hip --tapir-hip-xnack=ignore %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix IGNORE
//
// IGNORE: error: invalid argument 'ignore' to -tapir-hip-xnack=
//
// -----------------------------------------------------------------------------
