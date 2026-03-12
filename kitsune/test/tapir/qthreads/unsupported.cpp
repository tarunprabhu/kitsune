// The qthreads runtime only supports some operating system and architectures.
// Obviously we can't check all possible invalid targets, so just check that
// the error is triggered on a few that we know will not work. More importantly,
// check that it does not fail on targets that are known to be supported.
//
// RUN: %if x86-registered-target %{ \
// RUN:   %kitxx --tapir=qthreads --target=x86_64-pc-linux-gnu \
// RUN:       -c -O2 %s 2>&1 \
// RUN:       | FileCheck --allow-empty -check-prefix SUPPORTED %s \
// RUN: %}
//
// RUN: %if x86-registered-target %{ \
// RUN:   %kitxx --tapir=qthreads --target=x86_64-apple-macosx \
// RUN:       -c -O2 %s 2>&1 \
// RUN:       | FileCheck --allow-empty -check-prefix SUPPORTED %s \
// RUN: %}
//
// RUN: %if x86-registered-target %{ \
// RUN:   not %kitxx --tapir=qthreads --target=x86_64-pc-freebsd \
// RUN:       -c -O2 %s 2>&1 \
// RUN:       | FileCheck --allow-empty -check-prefix PLATFORM %s \
// RUN: %}
//
// RUN: %if x86-registered-target %{ \
// RUN:   not %kitxx --tapir=qthreads --target=x86_64-pc-openbsd \
// RUN:       -c -O2 %s 2>&1 \
// RUN:       | FileCheck --allow-empty -check-prefix PLATFORM %s \
// RUN: %}
//
// RUN: %if aarch64-registered-target %{ \
// RUN:   %kitxx --tapir=qthreads --target=aarch64-pc-linux-gnu %s \
// RUN:       -c -O2 %s 2>&1 \
// RUN:       | FileCheck --allow-empty -check-prefix SUPPORTED %s \
// RUN: %}
//
// RUN: %if sparc-registered-target %{\
// RUN:   not %kitxx --tapir=qthreads --target=sparc-pc-linux-gnu %s \
// RUN:       -c -O2 %s 2>&1 \
// RUN:       | FileCheck --allow-empty -check-prefix ARCH %s \
// RUN: %}
//
// ARCH: 'qthreads' tapir target does not support architecture
// PLATFORM: 'qthreads' tapir target does not support system
// SUPPORTED-NOT: 'qthreads' tapir target does not support
