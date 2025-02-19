// Providing a valid value to ftapir should not produce any output and return
// a success code. This only tests those backends that are always built.
//
// RUN: %kitxx -ftapir=none -fsyntax-only %s
// RUN: %kitxx -ftapir=serial -fsyntax-only %s

// The -ftapir flag is case sensitive.
// RUN: not %kitxx -ftapir=None -fsyntax-only %s 2>&1 | FileCheck %s -check-prefix BAD-TARGET
// RUN: not %kitxx -ftapir=Serial -fsyntax-only %s 2>&1 | FileCheck %s -check-prefix BAD-TARGET

// Unknown target
// RUN: not %kitxx -ftapir=fancy-target -fsyntax-only %s 2>&1 | FileCheck %s -check-prefix BAD-TARGET

// Last_TapirTargetID is a sentinel enum used in the code to indicate an
// "invalid" target. Test that this doesn't "leak".
// RUN: not %kitxx -ftapir=Last_TapirTargetID -fsyntax-only %s 2>&1 | FileCheck %s -check-prefix BAD-TARGET

// off used to be a valid value for the -ftapir flag, but it is not any longer.
// RUN: not %kitxx -ftapir=off -fsyntax-only %s 2>&1 | FileCheck %s -check-prefix BAD-TARGET
// RUN: not %kitxx -ftapir=Off -fsyntax-only %s 2>&1 | FileCheck %s -check-prefix BAD-TARGET

// The -ftapir option must be used with a Kitsune frontend.
// RUN: not %clang -ftapir=serial %s 2>&1 | FileCheck %s -check-prefix FRONTEND

// BAD-TARGET: invalid value '{{.+}}' in '-ftapir={{.+}}'
// FRONTEND: option '-ftapir=' must be used with a Kitsune frontend
