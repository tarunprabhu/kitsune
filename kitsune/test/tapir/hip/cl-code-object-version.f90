! REQUIRES: kitfc
!
! -----------------------------------------------------------------------------
! Check that the -mcode-object-version option is handled correctly. In our
! case, we only care that the correct bitcode file gets added to the list of
! bitcode files. Error handling when given invalid values are deferred to the
! driver.
!
! -----------------------------------------------------------------------------
! Ensure that the -mcode-object-version option is not actually required.
!
! RUN: %kitfc -### --tapir=hip -O2 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=DEFAULT
!
! DEFAULT: --tapir-hip-runtime-bcs={{.+}}/oclc_abi_version_{{[0-9]+}}.bc{{[^"]*}}"
! -----------------------------------------------------------------------------
! Ensure that invalid/unsupported values of the object version raise an error.
!
! RUN: not %kitfc -### --tapir=hip -mcode-object-version=3 -O2 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=ERROR_SUPPORT
!
! RUN: not %kitfc -### --tapir=hip --tapir-hip-abi-version=3 -O2 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=ERROR_SUPPORT
!
! RUN: not %kitfc -### --tapir=hip -mcode-object-version=4 -O2 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=ERROR_SUPPORT
!
! RUN: not %kitfc -### --tapir=hip --tapir-hip-abi-version=4 -O2 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=ERROR_SUPPORT
!
! RUN: not %kitfc -### --tapir=hip -mcode-object-version=99 -O2 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=ERROR_RANGE
!
! RUN: not %kitfc -### --tapir=hip --tapir-hip-abi-version=99 -O2 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=ERROR_RANGE
!
! ERROR_SUPPORT: rocm code object version '{{[0-9]+}}' is not supported
! ERROR_RANGE: cannot find ROCm device library for ABI version 99
!
! -----------------------------------------------------------------------------
! Check the supported values for the -mcode-object-version option and it's
! aliases. This should be updated to test all supported - and unsupported -
! values.
!
! RUN: %kitfc -### --tapir=hip -mcode-object-version=5 -O2 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=VER_5
!
! RUN: %kitfc -### --tapir=hip --tapir-hip-abi-version=5 -O2 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=VER_5
!
! VER_5: --tapir-hip-runtime-bcs={{.+}}/oclc_abi_version_500.bc{{[^"]*}}"

end program
