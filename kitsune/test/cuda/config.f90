! REQUIRES: kitfc
!
! ------------------------------------------------------------------------------
! Check that the default target-specific configuration file is always found.
!
! RUN: %kitfc -### --tapir=cuda --tapir-cuda-arch=sm_80 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix=CHECK-DEFAULT-CONFIG
!
! ------------------------------------------------------------------------------
! Check that providing a custom config directory without a target-specific
! configuration file is ok.
!
! RUN: %kitfc -### --tapir=cuda --tapir-cuda-arch=sm_80 \
! RUN:     --config-kitsune-dir=%S %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix=CHECK-CUSTOM-NOEXIST
!
! ------------------------------------------------------------------------------
! Check that providing a custom config directory with a target-specific
! configuration file leads to the file being found and the contents used and
! the default options are preserved.
!
! RUN: %kitfc -### -ftapir=cuda --tapir-cuda-arch=sm_80 \
! RUN:     --config-kitsune-dir=%S/input %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix=CHECK-CUSTOM
!
! CHECK-DEFAULT-CONFIG: Configuration file: {{.*}}/cuda.cfg
! CHECK-CUSTOM-NOEXIST-NOT: Configuration file: {{.*}}/cuda.cfg
! CHECK-CUSTOM: Configuration file: {{.*}}/input/cuda.cfg
! CHECK-CUSTOM: "-fc1"
! CHECK-CUSTOM-SAME: "-D" "some_preprocessor_flag"
! CHECK-CUSTOM-SAME: "-Wsome_compiler_flag"
!
! It is a pain to check for the actual linker executable. There are far too
! many options depending on the platform, so just check the next line for the
! expected linker flags.
!
! CHECK-CUSTOM-NEXT: "-some_linker_flag"
! CHECK-CUSTOM-SAME: -lkitrt
! CHECK-CUSTOM-SAME: -lcudart_static
! CHECK-CUSTOM-SAME: -lcuda
