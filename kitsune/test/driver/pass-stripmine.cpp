// Check that the stripmine pass is enabled/disabled as expected. This checks
// that the pipeline tuning object is setup correctly. This requires a tapir
// target, so we just use 'serial' since it is guaranteed to be available.
//
// TODO: It may be better to move these into the tests for individual tapir
// targets since the code that this exercises could change in a way that makes
// it more conditional on the tapir target.
//
// RUN: %kitxx -mllvm -print-pipeline-passes -O2 -fstripmine --tapir=serial \
// RUN:     -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix STRIPMINE-PASS
//
// RUN: %kitxx -mllvm -print-pipeline-passes -O2 -fno-stripmine --tapir=serial \
// RUN:     -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix NO-STRIPMINE-PASS
//
// STRIPMINE-PASS: loop-stripmine
// NO-STRIPMINE-PASS-NOT: loop-stripmine
