// -----------------------------------------------------------------------------
// The 'custom' tapir target requires the --tapir-plugin option.
//
// RUN: not %kitxx -### --tapir=custom %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=PLUGIN-MISSING
//
// PLUGIN-MISSING: error: '--tapir-plugin' is required
//
// -----------------------------------------------------------------------------
// The --tapir-plugin option requires a value
//
// RUN: not %kitxx -### --tapir=custom --tapir-plugin= -O1 %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=NOVAL
//
// NOVAL: error: argument to '--tapir-plugin=' is missing (expected 1 value)
//
// -----------------------------------------------------------------------------
// The --tapir-plugin option may be provided more than once, but all values
// must be identical.
//
// RUN: %kitxx -### --tapir=custom -O1 %s \
// RUN:     --tapir-plugin=plugin.ext --tapir-plugin=plugin.ext 2>&1 \
// RUN:     | FileCheck %s --check-prefix=UNIQ
//
// UNIQ: -cc1
// UNIQ-SAME: --tapir-plugin=plugin.ext
//
// -----------------------------------------------------------------------------
// If the --tapir-plugin option is provided more than once with different
// values, check that the expected diagnostic is emitted
//
// RUN: not %kitxx -### --tapir=custom -O1 %s \
// RUN:     --tapir-plugin=plugin1.ext --tapir-plugin=plugin2.ext 2>&1 \
// RUN:     | FileCheck %s --check-prefix=MULTIPLE
//
// MULTIPLE: error: '--tapir-plugin' must have a unique value
//
// -----------------------------------------------------------------------------
// The --tapir-plugin option requires --tapir=custom
//
// RUN: not %kitxx -### --tapir-plugin=plugin.ext -O1 %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=WRONG
//
// RUN: not %kitxx -### --tapir-plugin=plugin.ext --tapir=serial -O1 %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=WRONG
//
// WRONG: error: '--tapir-plugin' requires '--tapir=custom'
//
// -----------------------------------------------------------------------------
