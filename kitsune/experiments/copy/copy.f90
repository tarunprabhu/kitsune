program main
  implicit none

  integer(8) :: sz, iters
  integer(8) :: i, errors
  integer :: tick, tock, rate
  real(8) :: elapsed, lo, hi, total
  real(4), allocatable :: src(:), dst(:)

  sz = get_command_argument_or(1, lshift(1_8, 28_8))
  iters = get_command_argument_or(2, 10_8)

  write (*,"(A,I0,A/)") "  Problem size: ", sz, " elements"
  write (*,"(A)", advance="no") "  Allocating arrays and filling with random values ... "

  allocate(src(sz))
  allocate(dst(sz))

  write (*,"(A/)") "done"

  ! Initialize only the source because there is no need to initialize the
  ! destinations.
  call random_number(src)

  lo = huge(lo)
  hi = 0
  total = 0.0
  do i = 1, iters
     call system_clock(tick, rate)
     do concurrent (i = 1 : sz)
        dst(i) = src(i)
     end do
     call system_clock(tock, rate)
     elapsed = dble(tock - tick) / dble(rate)

     ! The first iteration is nearly always significantly slower than the
     ! others, probably because the cache needs to be warmed up.
     if (i > 1) then
        if (elapsed < lo) then
           lo = elapsed
        end if
        if (elapsed > hi) then
           hi = elapsed
        end if
        total = total + elapsed
     end if
     write(*, "(A,F10.3)") "  Iteration time: ", elapsed
  end do
  write(*, "(/A)", advance="no") "  Checking final result ... "

  errors = check(src, dst)
  if (errors /= 0) then
     write(*, "(A/)") "FAIL!"
  else
     write(*, "(A/)") "pass"
     write(*, "(A14,F10.3,A)") "  Total time: ", total, " seconds"
     write(*, "(A14,F10.3,A)") "    Min: ", lo, " seconds"
     write(*, "(A14,F10.3,A)") "    Max: ", hi, " seconds"
     write(*, "(A14,F10.3,A)") "    Mean: ", total / dble(iters - 1), " seconds"
  end if

  deallocate(dst)
  deallocate(src)

  if (errors /= 0) then
     stop errors
  end if

contains

  integer(8) function check(src, dst)
    implicit none

    real(4), allocatable, intent(in) :: src(:), dst(:)
    real(4), parameter :: epsilon = 1E-14
    integer(8) :: i

    check = 0
    do i = 1, size(src, 1)
       if (abs(dst(i) - src(i)) > epsilon) then
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
