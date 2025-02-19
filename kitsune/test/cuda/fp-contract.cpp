// The default FP contract value is different from clang's defaults. The new
// default is only used if a tapir target is specified.

// Check that the defaults have not changed when running without a Kitsune
// frontend
//
// RUN: %clang -### -x cuda %s 2>&1 | FileCheck %s -check-prefix DEFAULT-CUDA

// When running with a Kitsune frontend, this value should be ON. Also check
// that the value can be overridden if required.
//
// RUN: %kitxx -### %s 2>&1 | FileCheck %s -check-prefix CONTRACT-ON
// RUN: %kitxx -### -ftapir=cuda %s 2>&1 | FileCheck %s -check-prefix CONTRACT-ON
//
// RUN: %kitxx -### -ffp-contract=off %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix CONTRACT-OFF
// RUN: %kitxx -### -ffp-contract=off -ftapir=cuda %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix CONTRACT-OFF
//
// RUN: %kitxx -### -ffp-contract=fast %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix CONTRACT-FAST
// RUN: %kitxx -### -ffp-contract=fast -ftapir=cuda %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix CONTRACT-FAST

// CONTRACT-OFF: "-ffp-contract=off"
// CONTRACT-ON: "-ffp-contract=on"
// CONTRACT-FAST: "-ffp-contract=fast"

// When compiling for cuda, the host and device code are compiled separately
// The device code compilation (with a "primary" triple indicating nvptx) does
// not contain an ffp-contract entry - presumably because it is set to "fast"
// internally. On the host (with an "auxiliary" triple indicating nvptx), the
// fp-contract value remains set to the default of "ON".
//
// DEFAULT-CUDA: "-triple" "nvptx{{.*}}-nvidia-cuda"
// DEFAULT-CUDA-NOT: -ffp-contract
// DEFAULT-CUDA: "-aux-triple" "nvptx{{.*}}-nvidia-cuda"
// DEFAULT-CUDA-SAME: "-ffp-contract=on"
