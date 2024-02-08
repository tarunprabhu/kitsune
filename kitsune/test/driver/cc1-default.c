// Check that the default options added to the internal command lines (for -cc1
// and the linker) are as expected. There are corresponding tests that are
// tapir-target specific. Those generally check that the external libraries
// needed by that specific tapir target are linked. This is intended to check
// that libraries that are required by the "non-tapir-target-specific" parts of
// Kitsune's runtime are linked correctly.
//
// While this test is intended to be "independent" of a specific tapir targets,
// we cannot actually test it without using a tapir target. We pick 'serial'
// because:
//
//   1. It is guaranteed to be available
//   2. It does not require any external libraries, but does use libkitrt, so
//      it is reasonable to check for everything here.
//
// We could have added this to the test in transforms/tapir/serial, but that
// would make the intent of the test less clear.
//
// RUN: %kitcc -### --tapir=serial -O2 %s 2>&1 | FileCheck %s
//
// CHECK: -cc1
// CHECK-SAME: --tapir=serial
//
// We check for the absence of certain libraries that used to be linked
// explicitly in the past, but are not any longer. Calls to functions provided
// by these libraries should not be added directly by any lowering passes.
// Instead, a wrapper should be provided in libkitrt, and that should be called.
//
// CHECK-NOT: "-ldl"
// CHECK-NOT: "-lm"
// CHECK-NOT: "-lpthread"
// CHECK-NOT: "-lrt"
//
// The next line is expected to be the linker invocation. Since it is difficult
// to reliably check the name of the linker executable, just check for the
// expected linker flags. For C, we should not link a C++ standard library. If
// libkitrt requires it, it should be pulled in by that library.
//
// CHECK-NEXT: "-lkitrt"
// CHECK-NOT: "-l{{[^"]*}}c++"
