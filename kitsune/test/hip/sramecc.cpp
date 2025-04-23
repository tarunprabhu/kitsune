// Check that valid and invalid values of the --tapir-hip-sramecc option are
// handled correctly.
//
// -----------------------------------------------------------------------------
//
// RUN: %kitxx -### --tapir=hip --tapir-hip-sramecc=on %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix ON
//
// ON: -cc1
// ON-SAME: --tapir-hip-sramecc=on
//
// -----------------------------------------------------------------------------
//
// RUN: %kitxx -### --tapir=hip --tapir-hip-sramecc=off %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix OFF
//
// OFF: -cc1
// OFF-SAME: --tapir-hip-sramecc=off
//
// -----------------------------------------------------------------------------
//
// RUN: %kitxx -### --tapir=hip --tapir-hip-sramecc=any %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix ANY
//
// ANY: -cc1
// ANY: --tapir-hip-sramecc=any
//
// -----------------------------------------------------------------------------
//
// RUN: not %kitxx -### --tapir=hip --tapir-hip-sramecc= %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix MISSING
//
// MISSING: error: argument to '--tapir-hip-sramecc=' is missing
//
// -----------------------------------------------------------------------------
//
// RUN: not %kitxx -### --tapir=hip --tapir-hip-sramecc=ignore %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix IGNORE
//
// IGNORE: error: invalid argument 'ignore' to -tapir-hip-sramecc=
//
// -----------------------------------------------------------------------------
