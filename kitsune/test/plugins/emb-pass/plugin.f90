! REQUIRES: kitsune-examples, kitfc
!
! FIXME: Fortran support is as yet incomplete. When it is integrated in, we
! this should pass, but some changes, especially to the check lines may be
! required.
! XFAIL: *
!
! Check that the embedded module passes in a pass plugin are registered and
! run as expected.
!
! NOTE: The only defined function that will be printed will be the "kernel
! function" - consisting of the body of the tapir loop. We do not check for the
! precise name because it is not guaranteed to be consistent.
!
! RUN: %if kitsune-cuda %{ \
! RUN:   %kitfc --tapir=cuda --tapir-cuda-arch=sm_86 \
! RUN:       -O1 -S -emit-llvm -o /dev/null %s \
! RUN:       -fpass-plugin=%kit-emb-pass-plugin-demo 2>&1 \
! RUN:       | FileCheck %s \
! RUN: %}
!
! RUN: %if kitsune-hip %{ \
! RUN:   %kitfc --tapir=hip --tapir-hip-arch=gfx90a \
! RUN:       -O1 -S -emit-llvm -o /dev/null %s \
! RUN:       -fpass-plugin=%kit-emb-pass-plugin-demo 2>&1 \
! RUN:       | FileCheck %s \
! RUN: %}
!
! CHECK-DAG: declare external_func
! CHECK-DAG: define {{[^ ]+}}
! CHECK-DAG: declare llvm.kit.gpu.thread.id.x
! CHECK-DAG: declare llvm.kit.gpu.block.id.x
! CHECK-DAG: declare llvm.kit.gpu.block.size.x

module m

interface
  integer(8) pure function external_func(i)
    integer(8), value :: i
  end function external_func
end interface

contains
  subroutine mset(a, n)
    implicit none
    integer(8), allocatable :: a(:)
    integer(8) :: i, n

    do concurrent (i = 1 : n)
      a(i) = external_func(i)
    end do
  end subroutine mset

end module
