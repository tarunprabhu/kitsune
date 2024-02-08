module add

contains

  subroutine vecadd(a, b, c)
    implicit none
    integer(4), allocatable :: a(:), b(:), c(:)

    do concurrent (i = 1 : size(a))
      c(i) = a(i) + b(i)
    end do
  end subroutine vecadd
end module add
