// -----------------------------------------------------------------------------
// Providing a valid value to ftapir should not produce any output and return
// a success code. This only tests those backends that are always built.
//
// RUN: %kitxx -### -ftapir=none %s
// RUN: %kitxx -### --tapir=none %s
// RUN: %kitxx -### -ftapir=serial %s
// RUN: %kitxx -### -ftapir=serial %s

// -----------------------------------------------------------------------------
// The -ftapir flag is case sensitive.
//
// RUN: not %kitxx -### -ftapir=None %s 2>&1 | FileCheck %s -check-prefix BAD
// RUN: not %kitxx -### --tapir=Serial %s 2>&1 | FileCheck %s -check-prefix BAD

// Unknown target.
//
// RUN: not %kitxx -ftapir=fancy-target -fsyntax-only %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix BAD

// Last_TapirTargetID is a sentinel enum used in the code to indicate an
// "invalid" target. Test that this doesn't "leak".
//
// RUN: not %kitxx -ftapir=Last_TapirTargetID -fsyntax-only %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix BAD

// off used to be a valid value for the -ftapir flag, but it is not any longer.
//
// RUN: not %kitxx -### -ftapir=off %s 2>&1 | FileCheck %s -check-prefix BAD
// RUN: not %kitxx -### -ftapir=Off %s 2>&1 | FileCheck %s -check-prefix BAD

// BAD: invalid value '{{.+}}' in '-{{.}}tapir={{.+}}'

// -----------------------------------------------------------------------------
// The -ftapir option must be used with a Kitsune frontend.
//
// RUN: not %clang -ftapir=serial %s 2>&1 | FileCheck %s -check-prefix FRONTEND
// RUN: not %clang --tapir=serial %s 2>&1 | FileCheck %s -check-prefix FRONTEND

// FRONTEND: option '-{{.}}tapir=' must be used with a Kitsune frontend

// -----------------------------------------------------------------------------
// The -ftapir and --tapir options must be joined to the argument.
//
// RUN: not %kitxx -### -ftapir serial %s 2>&1 | FileCheck %s -check-prefix SPLIT
// RUN: not %kitxx -### --tapir serial %s 2>&1 | FileCheck %s -check-prefix SPLIT
//
// SPLIT: error: unknown argument: '-{{.}}tapir'
