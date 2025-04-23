// Check that an explicitly specified option is handled correctly
//
// RUN: %kitxx -### -ffp-contract=off -ftapir=serial %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix CONTRACT-OFF
//
// RUN: %kitxx -### -ffp-contract=on -ftapir=serial %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix CONTRACT-ON
//
// RUN: %kitxx -### -ffp-contract=fast -ftapir=serial %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix CONTRACT-FAST
//
// CONTRACT-OFF: "-ffp-contract=off"
// CONTRACT-ON: "-ffp-contract=on"
// CONTRACT-FAST: "-ffp-contract=fast"
