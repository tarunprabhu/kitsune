! The Kitsune=related options must must be used with a Kitsune frontend.

! RUN: %not %flang -ftapir=serial %s 2>&1 | FileCheck %s
! RUN: %not %flang -ftapir-cuda-arch=sm_80 %s 2>&1 | FileCheck %s
! RUN: %not %flang -ftapir-hip-arch=gfx90a %s 2>&1 | FileCheck %s

! CHECK: option '{{.+}}' must be used with a Kitsune frontend

end program
