// Kitsune requires that lld built alongside Kitsune be used. As a result,
// options that may override the linker are not allowed.
//
// RUN: not %kitxx --tapir=serial -O2 --ld-path=ld -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK: error: '--ld-path=' is not allowed in Kitsune
