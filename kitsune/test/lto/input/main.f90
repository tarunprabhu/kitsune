program main
  implicit none
  use add

  integer(8) :: sz = 128
  integer(4), allocatable :: a(:), b(:), c(:)

  allocate(a(sz))
  allocate(b(sz))
  allocate(c(sz))

  call vecadd(a, b, c)

  deallocate(c)
  deallocate(b)
  deallocate(a)

end program main
