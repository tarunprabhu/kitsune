// This has not been checked, so force it to fail if we ever resurrect this
// tapir target just so we are forced to take a look at this.
//
// RUN: false

// RUN: %kitxx -### -ftapir=qthreads %s 2>&1 | FileCheck %s

// CHECK: -cc1
// CHECK-SAME: -ftapir=qthreads

// It is a pain to check for the actual linker executable. There are far too
// many options depending on the platform, so just check the next line for the
// expected linker flags.
// CHECK: -lqthreads
// CHECK-SAME: -lkitrt
