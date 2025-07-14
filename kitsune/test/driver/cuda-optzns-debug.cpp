// REQUIRES: kitsune-cuda
//
// ptxas does not support optimized debugging. If optimizations and -g are both
// enabled, emit a warning that ptxas will be run at -O0, but ensure that the
// main optimization level remains unaffected.
//
// RUN: %kitxx -### --tapir=cuda -O1 -g %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes O1
//
// RUN: not %kitxx -### --tapir=cuda -O2 -g %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ERR
//
// RUN: not %kitxx -### --tapir=cuda -O3 -g %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ERR
//
// RUN: not %kitxx -### --tapir=cuda -Os -g %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ERR
//
// RUN: not %kitxx -### --tapir=cuda -Oz -g %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ERR
//
// O1: warning: ptxas does not support optimized debugging
// O1: -cc1
// O1-SAME: -O1
//
// ERR: error: ptxas does not support optimized debugging. Use -O1
