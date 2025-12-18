! REQUIRES: kitfc
!
! ------------------------------------------------------------------------------
! Check that providing a custom config directory without a target-specific
! configuration file is ok.
!
! RUN: %kitfc -### --tapir=cuda --tapir-cuda-arch=sm_80 -O1 \
! RUN:     --config-system-dir=%S %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix=CUSTOM-NOEXIST
!
! COM: %kitfc -### --tapir=cuda --tapir-cuda-arch=sm_80 -O1 \
! COM:     --config-user-dir=%S %s 2>&1 \
! COM:     | FileCheck %s -check-prefix=CUSTOM-NOEXIST
!
! CUSTOM-NOEXIST-NOT: Configuration file: {{.*}}/cuda.cfg
!
! ------------------------------------------------------------------------------
! Check that providing a custom config directory with a target-specific
! configuration file leads to the file being found and the contents used and
! the default options are preserved.
!
! RUN: %kitfc -### --tapir=cuda --tapir-cuda-arch=sm_80 -O1 \
! RUN:     --config-system-dir=%S/input %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix=CUSTOM
!
! RUN: %kitfc -### --tapir=cuda --tapir-cuda-arch=sm_80 -O1 \
! RUN:     --config-user-dir=%S/input %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix=CUSTOM
!
! CUSTOM: Configuration file: {{.*}}/input/cuda.cfg
! CUSTOM: "-fc1"
! CUSTOM-SAME: "-D" "some_preprocessor_flag"
! CUSTOM-SAME: "-Wsome_compiler_flag"
! CUSTOM-NEXT: "-some_linker_flag"
! CUSTOM-SAME: -lkitrt
! CUSTOM-SAME: -lcuda
! CUSTOM-SAME: -lcudart_static
!
! -----------------------------------------------------------------------------
! If configuration files for both the driver and the tapir target are present,
! check that the contents of both are used and the default options are
! preserved.
!
! Lit's configuration sets CLANG_NO_DEFAULT_CONFIG=1 before running these
! tests. With this set, the configuration files with the default names will not
! be read, even if a directory in which to look for configuration files has
! been explicitly provided. In most cases, this is exactly what we want, but
! here, we are explicitly checking that the default files are read, so this
! environment variable must be unset.
!
! RUN: env CLANG_NO_DEFAULT_CONFIG= \
! RUN: %kitfc -### --tapir=cuda --tapir-cuda-arch=sm_80 -O1 \
! RUN:     --config-system-dir=%S/input/cfgs %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix=BOTH
!
! RUN: env CLANG_NO_DEFAULT_CONFIG= \
! RUN: %kitfc -### --tapir=cuda --tapir-cuda-arch=sm_80 -O1 \
! RUN:     --config-user-dir=%S/input/cfgs %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix=BOTH
!
! BOTH: Configuration file: {{.*}}/input/cfgs/kitfc.cfg
! BOTH: Configuration file: {{.*}}/input/cfgs/cuda.cfg
! BOTH: "-fc1"
! BOTH-SAME: "-D" "driver_preprocessor_flag"
! BOTH-SAME: "-D" "tapir_preprocessor_flag"
! BOTH-SAME: "-Wdriver_compiler_flag"
! BOTH-SAME: "-Wtapir_compiler_flag"
! BOTH-NEXT: "-driver_linker_flag"
! BOTH-SAME: "-tapir_linker_flag"
! BOTH-SAME: -lkitrt
! BOTH-SAME: -lcuda
! BOTH-SAME: -lcudart_static
