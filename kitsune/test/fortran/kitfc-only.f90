! The Kitsune-related options must must be used with a Kitsune frontend.

! RUN: not %flang -### --tapir=serial %s 2>&1 | FileCheck %s
! RUN: not %flang -### --tapir-cuda-arch=sm_80 %s 2>&1 | FileCheck %s
! RUN: not %flang -### --tapir-hip-arch=gfx90a %s 2>&1 | FileCheck %s

! CHECK: option '{{.+}}' must be used with a Kitsune frontend

end program
