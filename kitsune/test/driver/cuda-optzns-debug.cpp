// REQUIRES: kitsune-cuda
//
// ptxas does not support optimized debugging. If optimizations and -g are both
// enabled, emit a warning that ptxas will be run at -O0, but ensure that the
// main optimization level remains unaffected.
//
// RUN: %kitxx -### --tapir=cuda --tapir-cuda-arch=sm_80 -O1 -g %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes O1
//
// RUN: not %kitxx -### --tapir=cuda --tapir-cuda-arch=sm_80 -O2 -g %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ERR
//
// RUN: not %kitxx -### --tapir=cuda -O3 --tapir-cuda-arch=sm_80 -g %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ERR
//
// RUN: not %kitxx -### --tapir=cuda -Os --tapir-cuda-arch=sm_80 -g %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ERR
//
// RUN: not %kitxx -### --tapir=cuda -Oz --tapir-cuda-arch=sm_80 -g %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes BADOPTLEVEL
//
// O1: warning: ptxas does not support optimized debugging
// O1: -cc1
// O1-SAME: -O1
//
// ERR: error: ptxas does not support optimized debugging. Use -O1
//
// BADOPTLEVEL: error: unsupported optimization level '-Oz'
