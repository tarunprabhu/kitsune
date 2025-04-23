// Check that valid and invalid values of the --tapir-hip-xnack option are
// handled correctly.
//
// -----------------------------------------------------------------------------
//
// RUN: %kitxx -### --tapir=hip --tapir-hip-xnack=on %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix ON
//
// ON: -cc1
// ON-SAME: --tapir-hip-xnack=on
//
// -----------------------------------------------------------------------------
//
// RUN: %kitxx -### --tapir=hip --tapir-hip-xnack=off %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix OFF
//
// OFF: -cc1
// OFF-SAME: --tapir-hip-xnack=off
//
// -----------------------------------------------------------------------------
//
// RUN: %kitxx -### --tapir=hip --tapir-hip-xnack=any %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix ANY
//
// ANY: -cc1
// ANY: --tapir-hip-xnack=any
//
// -----------------------------------------------------------------------------
//
// RUN: not %kitxx -### --tapir=hip --tapir-hip-xnack= %s 2>&1 \
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
