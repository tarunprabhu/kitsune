// The default FP contract value is different from clang's defaults. The new
// default is only used if a tapir target is specified.

// Check that the defaults have not changed when running without a Kitsune
// frontend
//
// RUN: %clang -### -x hip %s 2>&1 | FileCheck %s -check-prefix DEFAULT-HIP

// When running with a Kitsune frontend, this value should be ON. Also check
// that the value can be overridden if required.
//
// RUN: %kitxx -### %s 2>&1 | FileCheck %s -check-prefix CONTRACT-ON
// RUN: %kitxx -### -ftapir=hip %s 2>&1 | FileCheck %s -check-prefix CONTRACT-ON
//
// RUN: %kitxx -### -ffp-contract=off %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix CONTRACT-OFF
// RUN: %kitxx -### -ffp-contract=off -ftapir=hip %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix CONTRACT-OFF
//
// RUN: %kitxx -### -ffp-contract=fast %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix CONTRACT-FAST
// RUN: %kitxx -### -ffp-contract=fast -ftapir=hip %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix CONTRACT-FAST

// CONTRACT-OFF: "-ffp-contract=off"
// CONTRACT-ON: "-ffp-contract=on"
// CONTRACT-FAST: "-ffp-contract=fast"

// When compiling for cuda, the host and device code are compiled separately
// The device code compilation (with a "primary" triple indicating nvptx) does
// not contain an ffp-contract entry - presumably because it is set to "fast"
// internally. On the host (with an "auxiliary" triple indicating nvptx), the
// fp-contract value does not appear either. It is not clear from this test what
// it gets set to.
//
// DEFAULT-HIP: "-triple" "amd{{.*}}-amd-amdhsa"
// DEFAULT-HIP-NOT: -ffp-contract
// DEFAULT-HIP: "-aux-triple" "amd{{.*}}-amd-amdhsa"
// DEFAULT-HIP-NOT: "-ffp-contract=on"
//
// ----------------------------------------------------------------------------
//
// Check that the correct fp contact value is propagated to the runtime
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:      --tapir-verbose 2>&1 \
// RUN:      | FileCheck %s -check-prefix FUSION-STANDARD
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:      --tapir-verbose -ffp-contract=off 2>&1 \
// RUN:      | FileCheck %s -check-prefix FUSION-STANDARD
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:      --tapir-verbose -ffp-contract=on 2>&1 \
// RUN:      | FileCheck %s -check-prefix FUSION-STANDARD
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:      --tapir-verbose -ffp-contract=fast 2>&1 \
// RUN:      | FileCheck %s -check-prefix FUSION-FAST
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:      --tapir-verbose -ffp-contract=fast-honor-pragmas 2>&1 \
// RUN:      | FileCheck %s -check-prefix FUSION-STANDARD
//
// FUSION-STANDARD: FP Fusion: standard
// FUSION-FAST: FP Fusion: fast

#include <kitsune.h>

void f(int* c, size_t n) {
  forall(size_t i = 0; i < n; ++i)
    c[i] = n;
}
