! REQUIRES: kitsune-examples, kitfc
!
! The tapir target plugin should work with Fortran, but we currently do not have
! proper lowering of tapir loops. Until that is fixed, this will fail.
!
! XFAIL: *
!
! Check that a tapir target plugin works as expected on Fortran code. We use the
! tapir target plugin demo for consistency with the way LLVM pass plugins are
! tested.
!
! RUN: %kitfc --tapir=custom --tapir-plugin=%kit-ttplugin-demo %s \
! RUN:     -S -emit-llvm -o - -O2 \
! RUN:     | FileCheck %s --check-prefix=BOOKEND
!
! BOOKEND: call void @bookend
! BOOKEND-NEXT: call {{.*}}void @mset{{[^(]+}}(
! BOOKEND-NEXT: call void @bookend

module
contains
  subroutine mset(ptr, n)
    integer(8), allocatable :: ptr
    integer(8) :: n

    do i = 1, n
      ptr(i) = i
    end do
  end subroutine mset
end program
