// -----------------------------------------------------------------------------
// Check that the -mcode-object-version option is handled correctly. In our
// case, we only care that the correct bitcode file gets added to the list of
// bitcode files. Error handling when given invalid values are deferred to the
// driver.
//
// At the time of writing, only code object version 4 and 5 are available. It
// would be good if this could be updated to track all possible supported
// values.
//
// -----------------------------------------------------------------------------
// Ensure that the -mcode-object-version option is not actually required.
//
// RUN: %kitxx -### --tapir=hip -O2 %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=DEFAULT
//
// DEFAULT: --tapir-hip-runtime-bcs={{.+}}/oclc_abi_version_{{[0-9]+}}.bc{{[^"]*}}"
// -----------------------------------------------------------------------------
// Ensure that invalid/unsupported values of the object version raise an error.
//
// RUN: not %kitxx -### --tapir=hip -mcode-object-version=3 -O2 %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=ERROR_SUPPORT
//
// RUN: not %kitxx -### --tapir=hip --tapir-hip-abi-version=3 -O2 %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=ERROR_SUPPORT
//
// RUN: not %kitxx -### --tapir=hip -mcode-object-version=4 -O2 %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=ERROR_SUPPORT
//
// RUN: not %kitxx -### --tapir=hip --tapir-hip-abi-version=4 -O2 %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=ERROR_SUPPORT
//
// RUN: not %kitxx -### --tapir=hip -mcode-object-version=99 -O2 %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=ERROR_RANGE
//
// RUN: not %kitxx -### --tapir=hip --tapir-hip-abi-version=99 -O2 %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=ERROR_RANGE
//
// ERROR_SUPPORT: rocm code object version '{{[0-9]+}}' is not supported
// ERROR_RANGE: cannot find ROCm device library for ABI version 99
//
// -----------------------------------------------------------------------------
// Check the supported values for the -mcode-object-version option and it's
// aliases.
//
// RUN: %kitxx -### --tapir=hip -mcode-object-version=5 -O2 %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=VER_5
//
// RUN: %kitxx -### --tapir=hip --tapir-hip-abi-version=5 -O2 %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=VER_5
//
// VER_5: --tapir-hip-runtime-bcs={{.+}}/oclc_abi_version_500.bc{{[^"]*}}"
