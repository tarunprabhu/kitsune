// Check that the -mwavefrontsize64 and -mno-wavefrontsize64 options are handled
// correctly.
//
// In the tests here, provide the architecture of a GPU that does support both
// wavefront sizes of 32 and 64.
//
// RUN: %kitxx -### --tapir=hip --tapir-verbose -O2 -S -emit-llvm -o /dev/null \
// RUN:     --tapir-hip-arch=gfx1103 -mwavefrontsize64 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes W_64
//
// RUN: %kitxx -### --tapir=hip --tapir-verbose -O2 -S -emit-llvm -o /dev/null \
// RUN:     --tapir-hip-arch=gfx1103 --tapir-hip-wavefront64 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes W_64
//
// RUN: %kitxx -### --tapir=hip --tapir-verbose -O2 -S -emit-llvm -o /dev/null \
// RUN:     --tapir-hip-arch=gfx1103 -mno-wavefrontsize64 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes W_32
//
// RUN: %kitxx -### --tapir=hip --tapir-verbose -O2 -S -emit-llvm -o /dev/null \
// RUN:     --tapir-hip-arch=gfx1103 --tapir-hip-wavefront32 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes W_32
//
// Otherwise, if the GPU does not support a wavefront of 32, the
// -mno-wavefrontsize64 option will be ignored.
//
// RUN: %kitxx -### --tapir=hip --tapir-verbose -O2 -S -emit-llvm -o /dev/null \
// RUN:     --tapir-hip-arch=gfx90a -mwavefrontsize64 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes W_64
//
// RUN: %kitxx -### --tapir=hip --tapir-verbose -O2 -S -emit-llvm -o /dev/null \
// RUN:     --tapir-hip-arch=gfx90a --tapir-hip-wavefront64 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes W_64
//
// RUN: %kitxx -### --tapir=hip --tapir-verbose -O2 -S -emit-llvm -o /dev/null \
// RUN:     --tapir-hip-arch=gfx90a -mno-wavefrontsize64 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes W_64
//
// RUN: %kitxx -### --tapir=hip --tapir-verbose -O2 -S -emit-llvm -o /dev/null \
// RUN:     --tapir-hip-arch=gfx90a --tapir-hip-wavefront32 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes W_64
//
// W_64: --tapir-hip-features=
// W_64-SAME: +wavefrontsize64
// W_64-SAME: --tapir-hip-runtime-bcs={{.+}}/oclc_wavefrontsize64_on.bc{{[^"]*}}"
//
// W_32: --tapir-hip-features=
// W_32-SAME: +wavefrontsize32
// W_32-SAME: --tapir-hip-runtime-bcs={{.+}}/oclc_wavefrontsize64_off.bc{{[^"]*}}"
// -----------------------------------------------------------------------------
// If neither -mwavefront64, nor -mno-wavefront64 is provided, the wavefront
// size is determined by the GPU architecture.
//
// RUN: %kitxx -### --tapir=hip --tapir-verbose -O2 -S -emit-llvm -o /dev/null \
// RUN:     --tapir-hip-arch=gfx906 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes DEFAULT_64
//
// RUN: %kitxx -### --tapir=hip --tapir-verbose -O2 -S -emit-llvm -o /dev/null \
// RUN:     --tapir-hip-arch=gfx1103 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes DEFAULT_32
//
//
// DEFAULT_64: --tapir-hip-features={{.*}}+wavefrontsize64{{[^"]*}}"
// DEFAULT_64-SAME: --tapir-hip-runtime-bcs={{.+}}/oclc_wavefrontsize64_on.bc{{[^"]*}}"
//
// DEFAULT_32-NOT: +wavefrontsize64
// DEFAULT_32: --tapir-hip-runtime-bcs={{.+}}/oclc_wavefrontsize64_off.bc{{[^"]*}}"
