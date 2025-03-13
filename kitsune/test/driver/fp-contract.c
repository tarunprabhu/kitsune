// The default FP contract value is different from clang's defaults. The new
// default is only used if a tapir target is specified.
//
// Check that the defaults have not changed when running without a Kitsune
// frontend
//
// RUN: %clang -### %s 2>&1 | FileCheck %s -check-prefix CONTRACT-ON
//
// When running with a Kitsune frontend, this value should be ON. Also check
// that the value can be overridden if required.
//
// RUN: %kitcc -### %s 2>&1 | FileCheck %s -check-prefix CONTRACT-ON
// RUN: %kitcc -### -ffp-contract=off %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix CONTRACT-OFF
// RUN: %kitcc -### -ffp-contract=off -ftapir=serial %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix CONTRACT-OFF
// RUN: %kitcc -### -ffp-contract=fast %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix CONTRACT-FAST
// RUN: %kitcc -### -ffp-contract=fast -ftapir=serial %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix CONTRACT-FAST
//
// CONTRACT-OFF: "-ffp-contract=off"
// CONTRACT-ON: "-ffp-contract=on"
// CONTRACT-FAST: "-ffp-contract=fast"
