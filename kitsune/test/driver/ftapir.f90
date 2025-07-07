! REQUIRES: kitfc
!
! -----------------------------------------------------------------------------
! Providing a valid value to ftapir should not produce any output and return
! a success code. This only tests those backends that are always built.
!
! RUN: %kitfc -### -ftapir=nolo -O1 %s
! RUN: %kitfc -### --tapir=nolo -O1 %s
! RUN: %kitfc -### -ftapir=serial %s -O1
! RUN: %kitfc -### --tapir=serial %s -O1
!
! -----------------------------------------------------------------------------
! The -ftapir flag is case sensitive.
!
! RUN: not %kitfc -### -ftapir=None %s -O1 2>&1 \
! RUN:     | FileCheck %s -check-prefix BAD
! RUN: not %kitfc -### --tapir=Serial %s -O1 2>&1 \
! RUN:     | FileCheck %s -check-prefix BAD
!
! -----------------------------------------------------------------------------
! Unknown target.
!
! RUN: not %kitfc -ftapir=fancy-target -fsyntax-only %s -O1 2>&1 \
! RUN:     | FileCheck %s -check-prefix BAD
!
! off used to be a valid value for the -ftapir flag, but it is not any longer.
!
! RUN: not %kitfc -### --tapir=off %s -O1 2>&1 | FileCheck %s -check-prefix BAD
! RUN: not %kitfc -### --tapir=Off %s -O1 2>&1 | FileCheck %s -check-prefix BAD
!
! BAD: invalid value '{{.+}}' in '-{{.}}tapir={{.+}}'
!
! -----------------------------------------------------------------------------
! The -ftapir option must be used with a Kitsune frontend.
!
! RUN: not %flang -ftapir=serial %s -O1 2>&1 \
! RUN:     | FileCheck %s -check-prefix FRONTEND
! RUN: not %flang --tapir=serial %s -O1 2>&1 \
! RUN:     | FileCheck %s -check-prefix FRONTEND
!
! FRONTEND: option '-{{.}}tapir=' must be used with a Kitsune frontend
!
! -----------------------------------------------------------------------------
! The -ftapir and --tapir options must be joined to the argument.
!
! RUN: not %kitfc -### -ftapir serial %s -O1 2>&1 \
! RUN:     | FileCheck %s -check-prefix SPLIT
! RUN: not %kitfc -### --tapir serial %s -O1 2>&1 \
! RUN:     | FileCheck %s -check-prefix SPLIT
!
! SPLIT: error: unknown argument: '-{{.}}tapir'
