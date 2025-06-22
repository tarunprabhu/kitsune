! REQUIRES: kitfc
!
! The opencilk runtime only supports some operating system and architectures.
! Obviously we can't check all possible invalid targets, so just check that
! the error is triggered on a few that we know will not work. More importantly,
! check that it does not fail on targets that are known to be supported.
!
! RUN: %if x86-registered-target %{ \
! RUN:   %kitfc --tapir=opencilk --target=x86_64-unknown-linux-gnu \
! RUN:       -Xflang --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc \
! RUN:       -c -O2 %s 2>&1 \
! RUN:       | FileCheck --allow-empty -check-prefix SUPPORTED %s \
! RUN: %}
!
! RUN: %if x86-registered-target %{ \
! RUN:   %kitfc --tapir=opencilk --target=x86_64-pc-freebsd \
! RUN:       -Xflang --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc \
! RUN:       -c -O2 %s 2>&1 \
! RUN:       | FileCheck --allow-empty -check-prefix SUPPORTED %s \
! RUN: %}
!
! RUN: %if x86-registered-target %{ \
! RUN:   %kitfc --tapir=opencilk --target=x86_64-apple-macosx \
! RUN:       -Xflang --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc \
! RUN:       -c -O2 %s 2>&1 \
! RUN:       | FileCheck --allow-empty -check-prefix SUPPORTED %s \
! RUN: %}
!
! RUN: %if x86-registered-target %{ \
! RUN:   not %kitfc --tapir=opencilk --target=x86_64-pc-openbsd \
! RUN:       -Xflang --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc \
! RUN:       -c -O2 %s 2>&1 \
! RUN:       | FileCheck --allow-empty -check-prefix PLATFORM %s \
! RUN: %}
!
! RUN: %if aarch64-registered-target %{ \
! RUN:   %kitfc --tapir=opencilk --target=aarch64-unknown-linux-gnu \
! RUN:       -Xflang --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc \
! RUN:       -c -O2 %s 2>&1 \
! RUN:       | FileCheck --allow-empty -check-prefix SUPPORTED %s \
! RUN: %}
!
! RUN: %if sparc-registered-target %{\
! RUN:   not %kitfc --tapir=opencilk --target=sparc-pc-linux-gnu \
! RUN:       -Xflang --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc \
! RUN:       -c -O2 %s 2>&1 \
! RUN:       | FileCheck --allow-empty -check-prefix ARCH %s \
! RUN: %}
!
! ARCH: opencilk tapir target does not support architecture
! PLATFORM: opencilk tapir target does not support system
! SUPPORTED-NOT: opencilk tapir target does not support
