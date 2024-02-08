// The opencilk runtime only supports some operating system and architectures.
// Obviously we can't check all possible invalid targets, so just check that
// the error is triggered on a few that we know will not work. More importantly,
// check that it does not fail on targets that are known to be supported.

// RUN: %if x86-registered-target %{ \
// RUN:   %kitxx --tapir=opencilk --target=x86_64-unknown-linux-gnu -c -O2 %s \
// RUN:       -Xclang --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc \
// RUN:        2>&1 | FileCheck --allow-empty -check-prefix SUPPORTED %s \
// RUN: %}

// RUN: %if x86-registered-target %{ \
// RUN:   %kitxx --tapir=opencilk --target=x86_64-pc-freebsd -c -O2 %s \
// RUN:       -Xclang --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc \
// RUN:        2>&1 | FileCheck --allow-empty -check-prefix SUPPORTED %s \
// RUN: %}

// RUN: %if x86-registered-target %{ \
// RUN:   %kitxx --tapir=opencilk --target=x86_64-apple-macosx -c -O2 %s \
// RUN:       -Xclang --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc \
// RUN:        2>&1 | FileCheck --allow-empty -check-prefix SUPPORTED %s \
// RUN: %}

// RUN: %if x86-registered-target %{ \
// RUN:   not %kitxx --tapir=opencilk --target=x86_64-pc-openbsd -c -O2 %s \
// RUN:       -Xclang --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc \
// RUN:        2>&1 | FileCheck --allow-empty -check-prefix PLATFORM %s \
// RUN: %}

// RUN: %if aarch64-registered-target %{ \
// RUN:   %kitxx --tapir=opencilk --target=aarch64-unknown-linux-gnu -c -O2 %s \
// RUN:       -Xclang --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc \
// RUN:        2>&1 | FileCheck --allow-empty -check-prefix SUPPORTED %s \
// RUN: %}

// RUN: %if sparc-registered-target %{\
// RUN:   not %kitxx --tapir=opencilk --target=sparc-pc-linux-gnu -c -O2 %s \
// RUN:       -Xclang --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc \
// RUN:        2>&1 | FileCheck --allow-empty -check-prefix ARCH %s \
// RUN: %}

// ARCH: opencilk tapir target does not support architecture
// PLATFORM: opencilk tapir target does not support system
// SUPPORTED-NOT: opencilk tapir target does not support
