program main
  implicit none

  integer(8) :: m, n, k, iters
  integer(8) :: i, j, l, errors
  integer :: tick, tock, rate
  real(8) :: elapsed, lo, hi, total
  real(4), allocatable :: a(:), b(:), c(:)
  real(4) :: sum

  m = get_command_argument_or(1, 512_8)
  n = get_command_argument_or(2, 512_8)
  k = get_command_argument_or(3, 512_8)
  iters = get_command_argument_or(4, 10_8)

  write (*,"(A,I0,A,I0,A,I0/)") "  Problem size: ", m, " x ", n, " x ", k
  write (*,"(A)", advance="no") "  Allocating arrays and filling with random values ... "

  allocate(a(m * k))
  allocate(b(k * n))
  allocate(c(m * n))

  write (*,"(A/)") "done"

  ! Initialize only the operands, not the result.
  call random_number(a)
  call random_number(b)

  lo = huge(lo)
  hi = 0
  total = 0.0
  do l = 1, iters
     call system_clock(tick, rate)
     do concurrent (i = 0 : m * n - 1)
        sum = 0.0
        do j = 0, k - 1
           sum = sum + a((i / n) * k + j + 1) * b(rem(i, n) * k + j + 1)
        end do
        c(i + 1) = sum
     end do
     call system_clock(tock, rate)
     elapsed = dble(tock - tick) / dble(rate)

     if (elapsed < lo) then
        lo = elapsed
     end if
     if (elapsed > hi) then
        hi = elapsed
     end if
     total = total + elapsed
     write(*, "(A,F10.3)") "  Iteration time: ", elapsed
  end do
  write(*, "(/A)", advance="no") "  Checking final result ... "

  errors = check(a, b, c, m, n, k)
  if (errors /= 0) then
     write(*, "(A/)") "FAIL!"
  else
     write(*, "(A/)") "pass"
     write(*, "(A14,F10.3,A)") "  Total time: ", total, " seconds"
     write(*, "(A14,F10.3,A)") "    Min: ", lo, " seconds"
     write(*, "(A14,F10.3,A)") "    Max: ", hi, " seconds"
     write(*, "(A14,F10.3,A)") "    Mean: ", total / iters, " seconds"
  end if

  deallocate(c)
  deallocate(b)
  deallocate(a)

  if (errors /= 0) then
     stop errors
  end if

contains

  integer(8) pure function rem(m, n)
    implicit none
    integer(8), intent(in) :: m, n
    if (m < n) then
       rem = m
    else
       rem = mod(m, n)
    end if
  end function rem

  integer(8) function check(a, b, c, m, n, k)
    implicit none

    real(4), allocatable, intent(in) :: a(:), b(:), c(:)
    integer(8), intent(in) :: m, n, k
    real(4), parameter :: epsilon = 1E-14
    integer(8) :: i, j
    real(4) :: sum

    check = 0
    do i = 0, m * n - 1
       sum = 0.0
       do j = 0, k - 1
          sum = sum + a((i / n) * k + j + 1) * b(rem(i, n) * k + j + 1)
       end do
       if (abs(c(i + 1) - sum) > epsilon) then
          check = check + 1
       end if
    end do
  end function check

  integer(8) function get_command_argument_or(n, default)
    implicit none

    integer :: n
    integer(8) :: default
    character(32) :: buf

    if (command_argument_count() >= n) then
       call get_command_argument(n, buf)
       read (buf, *) get_command_argument_or
    else
       get_command_argument_or = default
    end if
  end function get_command_argument_or
end program main
