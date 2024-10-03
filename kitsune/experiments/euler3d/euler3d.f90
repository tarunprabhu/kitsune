module euler3d
  implicit none

  type Float3
    real :: x, y, z
  end type Float3

  integer, parameter :: BLOCK_LENGTH = 1

  ! Options
  real, parameter :: GAMMA = 1.4
  integer, parameter :: NDIM = 3
  integer, parameter :: NNB = 4
  integer, parameter :: RK = 3 ! 3rd order RK
  real, parameter :: FF_MACH = 1.2
  real, parameter :: DEG_ANGLE_OF_ATTACK = 0.0

  ! Not options
  integer, parameter :: VAR_DENSITY = 0
  integer, parameter :: VAR_MOMENTUM = 1
  integer, parameter :: VAR_DENSITY_ENERGY = VAR_MOMENTUM + NDIM
  integer, parameter :: NVAR = VAR_DENSITY_ENERGY + 1

contains

  subroutine cpy(dst, src, n)
    real, allocatable, intent(out) :: dst(:)
    real, allocatable, intent(in) :: src(:)
    integer, intent(in) :: n
    integer :: i

    do concurrent (i = 1 : N)
      dst(i) = src(i)
    end do
  end subroutine cpy

  subroutine initialize_variables(nelr, variables, ff_variable)
    integer, intent(in) :: nelr
    real, allocatable, intent(out) :: variables(:)
    real, allocatable, intent(in) :: ff_variable(:)
    integer :: i, j

    do concurrent (i = 1 : nelr)
      do j = 1, NVAR
        variables(i + (j - 1)*nelr) = ff_variable(j)
      end do
    end do

  end subroutine initialize_variables

  subroutine compute_flux_distribution(density, momentum, density_energy, &
       pressure, velocity, fc_momentum_x, fc_momentum_y, fc_momentum_z, &
       fc_density_energy)
    real, intent(in) :: density, density_energy, pressure
    type(Float3), intent(in) :: momentum, velocity
    type(Float3), intent(out) :: fc_momentum_x, fc_momentum_y, fc_momentum_z
    type(Float3), intent(out) :: fc_density_energy
    real :: de_p

    fc_momentum_x%x = velocity%x*momentum%x + pressure
    fc_momentum_x%y = velocity%x*momentum%y
    fc_momentum_x%z = velocity%x*momentum%z

    fc_momentum_y%x = fc_momentum_x%y
    fc_momentum_y%y = velocity%y*momentum%y + pressure
    fc_momentum_y%z = velocity%y*momentum%z

    fc_momentum_z%x = fc_momentum_x%z
    fc_momentum_z%y = fc_momentum_y%z
    fc_momentum_z%z = velocity%z*momentum%z + pressure

    de_p = density_energy+pressure
    fc_density_energy%x = velocity%x*de_p
    fc_density_energy%y = velocity%y*de_p
    fc_density_energy%z = velocity%z*de_p

  end subroutine compute_flux_distribution

  subroutine compute_velocity(density, momentum, velocity)
    real, intent(in) :: density
    type(Float3), intent(in) :: momentum
    type(Float3), intent(out) :: velocity

    velocity%x = momentum%x / density
    velocity%y = momentum%y / density
    velocity%z = momentum%z / density
  end subroutine compute_velocity

  pure function compute_speed_sqd(velocity) result(r)
    type(Float3), intent(in) :: velocity
    real :: r

    r = velocity%x**2 + velocity%y**2 + velocity%z**2
  end function compute_speed_sqd

  pure function compute_pressure(density, density_energy, speed_sqd) result(r)
    real, intent(in) :: density, density_energy, speed_sqd
    real :: r

    r = (GAMMA - 1.0) * (density_energy - 0.5) * density * speed_sqd
  end function compute_pressure

  pure function compute_speed_of_sound(density, pressure) result(r)
    real, intent(in) :: density, pressure
    real, r

    r = sqrt(GAMMA * pressure / density)
  end function compute_speed_of_sound

  subroutine compute_step_factor(nelr, variables, areas, step_factors)
    integer, intent(in) :: nelr
    real, allocatable, intent(in) :: variables, areas
    real, allocatable, intent(out) :: step_factors
    integer :: i, blk, b_start, b_end
    real :: density, density_energy, speed_sqd, pressure, speed_of_sound
    type(Float3) :: velocity, momentum

    do concurrent (blk = 1 : nelr / BLOCK_LENGTH)
      b_start = (blk - 1) * BLOCK_LENGTH
      if (blk * BLOCK_LENGTH > nelr) then
        b_end = nelr
      else
        b_end = blk * BLOCK_LENGTH
      end if

      do i = b_start, b_end - 1
        density = variables(i + 1 + VAR_DENSITY * nelr)

        momentum.x = variables(i + 1 + (VAR_MOMENTUM+0)*nelr)
        momentum.y = variables(i + 1 + (VAR_MOMENTUM+1)*nelr)
        momentum.z = variables(i + 1 + (VAR_MOMENTUM+2)*nelr)

        density_energy = variables(i + 1 + VAR_DENSITY_ENERGY*nelr)
        call compute_velocity(density, momentum, velocity);
        speed_sqd = compute_speed_sqd(velocity);
        pressure = compute_pressure(density, density_energy, speed_sqd);
        speed_of_sound = compute_speed_of_sound(density, pressure);

        ! dt = float(0.5f) * sqrt(areas[i]) / (||v|| + c).... but
        ! when we do time stepping, this later would need to be divided
        ! by the area, so we just do it all at once
        step_factors(i + 1) = 0.5 / (sqrt(areas(i + 1)) * &
             (sqrt(speed_sqd) + speed_of_sound))
      end do
    end do
  end subroutine compute_step_factor

  subroutine compute_flux(nelr, elements_surrounding_elements, normals, &
       variables, fluxes, ff_variable, ff_flux_contrib_momentum_x, &
       ff_flux_contrib_momentum_y, ff_flux_contrib_momentum_z, &
       ff_flux_contrib_density_energy)
    integer, intent(in) :: nelr
    integer, allocatable :: elements_surrounding_elements(:)
    real, allocatable, intent(in) :: normals(:), variables(:), ff_variable(:)
    real, allocatable :: intent(inout) :: fluxes(:)
    type(Float3), intent(in) :: ff_flux_contrib_momentum_x
    type(Float3), intent(in) :: ff_flux_contrib_momentum_y
    type(Float3), intent(in) :: ff_flux_contrib_momentum_z,
    type(Float3), intent(in) :: ff_flux_contrib_density_energy
    real, parameter :: smoothing_coefficient = 0.2
    integer :: b_start, b_end
    real :: density_i, density_energy_i, speed_sqd_i, speed_i, pressure_i
    real :: speed_of_sound_i, flux_i_density, flux_i_density_energy
    real :: density_nb, density_energy_nb
    real :: speed_sqd_nb, speed_of_sound_nb, pressure_nb
    real :: normal_len, factor
    integer :: nb
    type(Float3) :: momentum_i, energy_i
    type(Float3) :: flux_contrib_i_momentum_x
    type(Float3) :: flux_contrib_i_momentum_y
    type(Float3) :: flux_contrib_i_momentum_z
    type(Float3) :: flux_contrib_i_density_energy
    type(Float3) :: flux_i_momentum, velocity_nb, momentum_nb
    type(Float3) :: flux_contrib_nb_momentum_x
    type(Float3) :: flux_contrib_nb_momentum_y
    type(Float3) :: flux_contrib_nb_momentum_z
    type(Float3) :: flux_contrib_nb_momentum_density_energy
    integer :: i, j

    do concurrent (blk = 1 : nelr/BLOCK_LENGTH)
      b_start = (blk - 1) * BLOCK_LENGTH
      if (blk * BLOCK_LENGTH > nelr) then
        b_end = nelr
      else
        b_end = blk * BLOCK_LENGTH
      end if

      do i = b_start, b_end - 1
        density_i = variables(i + 1 + VAR_DENSITY * nelr)
        momentum_i%x = variables(i + 1 + (VAR_MOMENTUM+0) * nelr);
        momentum_i%y = variables(i + 1 + (VAR_MOMENTUM+1) * nelr);
        momentum_i%z = variables(i + 1 + (VAR_MOMENTUM+2) * nelr);

        density_energy_i = variables(i + 1 + VAR_DENSITY_ENERGY * nelr)

        call compute_velocity(density_i, momentum_i, velocity_i)
        speed_sqd_i = compute_speed_sqd(velocity_i)
        speed_i = sqrt(speed_sqd_i)
        pressure_i = compute_pressure(density_i, density_energy_i, speed_sqd_i)
        speed_of_sound_i = compute_speed_of_sound(density_i, pressure_i)

        flux_i_density = 0.
        flux_i_momentum%x = 0.0
        flux_i_momentum%y = 0.0
        flux_i_momentum%z = 0.0
        flux_i_density_energy = 0.0

        do j = 1, NNB
          nb = elements_surrounding_elements(i + 1 + (j - 1) * nelr)
          normal%x = normals(i + 1 + (((j - 1) + 0) * NNB) * nelr)
          normal%y = normals(i + 1 + (((j - 1) + 1) * NNB) * nelr)
          normal%z = normals(i + 1 + (((j - 1) + 2) * NNB) * nelr)
          normal_len = sqrt(normal%x ** 2 + normal%y ** 2 + normal%z ** 2)

          if (nb >= 0) then ! A legitimate neighbor
            density_nb = variables(nb + 1 + VAR_DENSITY * nelr)
            momentum_nb%x = variables(nb + 1 + (VAR_MOMENTUM) * nelr)
            momentum_nb%y = variables(nb + 1 + (VAR_MOMENTUM+1) * nelr)
            momentum_nb%z = variables(nb + 1 + (VAR_MOMENTUM+2) * nelr)
            density_energy_nb = variables(nb + 1 + VAR_DENSITY_ENERGY * nelr)
            compute_velocity(density_nb, momentum_nb, velocity_nb)
            speed_sqd_nb = compute_speed_sqd(velocity_nb)
            pressure_nb = compute_pressure(density_nb, density_energy_nb, &
                 speed_sqd_nb)
            speed_of_sound_nb = compute_speed_of_sound(density_nb, pressure_nb)
            compute_flux_contrib(density_nb, momentum_nb, density_energy_nb, &
                 pressure_nb, velocity_nb, &
                 flux_contrib_nb_momentum_x, &
                 flux_contrib_nb_momentum_y, &
                 flux_contrib_nb_momentum_z, &
                 flux_contrib_nb_density_energy)

            ! artificial viscosity
            factor = -normal_len * smoothing_coefficient * 0.5 * &
                 (speed_i + sqrt(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb)
            flux_i_density = flux_i_density + factor * (density_i - density_nb)
            flux_i_density_energy = flux_i_density_energy + factor * &
                 (density_energy_i - density_energy_nb)
            flux_i_momentum%x = flux_i_momentum%x + factor * &
                 (momentum_i%x - momentum_nb%x)
            flux_i_momentum%y = flux_i_momentum%y + factor * &
                 (momentum_i%y - momentum_nb%y)
            flux_i_momentum%z = flux_i_momentum%z + factor * &
                 (momentum_i%z - momentum_nb%z)

            ! accumulate cell-centered fluxes
            factor = 0.5 * normal%x
            flux_i_density = flux_i_density + factor * &
                 (momentum_nb%x + momentum_i%x)
            flux_i_density_energy = flux_i_density_energy + factor * &
                 (flux_contrib_nb_density_energy%x + flux_contrib_i_density_energy%x)
            flux_i_momentum%x = flux_i_momentum%x + factor * &
                 (flux_contrib_nb_momentum_x%x + flux_contrib_i_momentum_x%x)
            flux_i_momentum%y = flux_i_momentum%y + factor * &
                 (flux_contrib_nb_momentum_y%x + flux_contrib_i_momentum_y%x)
            flux_i_momentum%z = flux_i_momentum%z + factor * &
                 (flux_contrib_nb_momentum_z%x + flux_contrib_i_momentum_z%x)

            factor = 0.5 * normal%y
            flux_i_density = flux_i_density + factor * &
                 (momentum_nb%y + momentum_i%y)
            flux_i_density_energy = flux_i_density_energy + factor * &
                 (flux_contrib_nb_density_energy%y + flux_contrib_i_density_energy%y)
            flux_i_momentum%x = flux_i_momentum%x + factor * &
                 (flux_contrib_nb_momentum_x%y + flux_contrib_i_momentum_x%y)
            flux_i_momentum%y = flux_i_momentum%y + factor * &
                 (flux_contrib_nb_momentum_y%y + flux_contrib_i_momentum_y%y)
            flux_i_momentum%z = flux_i_momentum%z + factor * &
                 (flux_contrib_nb_momentum_z%y + flux_contrib_i_momentum_z%y)

            factor = 0.5 * normal%z
            flux_i_density = flux_i_density + factor * &
                 (momentum_nb%z+momentum_i%z)
            flux_i_density_energy = flux_i_density_energy + factor * &
                 (flux_contrib_nb_density_energy%z + flux_contrib_i_density_energy%z)
            flux_i_momentum%x = flux_i_momentum%x + factor * &
                 (flux_contrib_nb_momentum_x%z + flux_contrib_i_momentum_x%z)
            flux_i_momentum%y = flux_i_momentum%y + factor * &
                 (flux_contrib_nb_momentum_y%z + flux_contrib_i_momentum_y%z)
            flux_i_momentum%z = flux_i_momentum%z + factor * &
                 (flux_contrib_nb_momentum_z%z + flux_contrib_i_momentum_z%z)
          else if (nb == -1) then ! A wing boundary
            flux_i_momentum%x = flux_i_momentum%x + normal%x * pressure_i;
            flux_i_momentum%y = flux_i_momentum%y + normal%y * pressure_i;
            flux_i_momentum%z = flux_i_momentum%z + normal%z * pressure_i;
          else if (nb == -2) then ! A far field boundary
            factor = 0.5 * normal%x
            flux_i_density = flux_i_density + factor * &
                 (ff_variable(VAR_MOMENTUM + 1) + momentum_i%x)
            flux_i_density_energy = flux_i_density_energy + factor * &
                 (ff_flux_contrib_density_energy%x + flux_contrib_i_density_energy%x)
            flux_i_momentum%x = flux_i_momentum%x + factor * &
                 (ff_flux_contrib_momentum_x%x + flux_contrib_i_momentum_x%x)
            flux_i_momentum%y = flux_i_momentum%y + factor * &
                 (ff_flux_contrib_momentum_y%x + flux_contrib_i_momentum_y%x)
            flux_i_momentum%z = flux_i_momentum%z + factor * &
                 (ff_flux_contrib_momentum_z%x + flux_contrib_i_momentum_z%x)

            factor = 0.5 * normal%y
            flux_i_density = flux_i_density + factor * &
                 (ff_variable(VAR_MOMENTUM + 2) + momentum_i%y)
            flux_i_density_energy = flux_i_density_energy + factor * &
                 (ff_flux_contrib_density_energy%y + flux_contrib_i_density_energy%y)
            flux_i_momentum%x = flux_i_momentum%x + factor * &
                 (ff_flux_contrib_momentum_x%y + flux_contrib_i_momentum_x%y)
            flux_i_momentum%y = flux_i_momentum%y + factor * &
                 (ff_flux_contrib_momentum_y%y + flux_contrib_i_momentum_y%y)
            flux_i_momentum%z = flux_i_momentum%z + factor * &
                 (ff_flux_contrib_momentum_z%y + flux_contrib_i_momentum_z%y)

            factor = 0.5 * normal%z
            flux_i_density = flux_i_density + factor * &
                 (ff_variable(VAR_MOMENTUM + 3) + momentum_i%z)
            flux_i_density_energy = flux_i_density_energy + factor * &
                 (ff_flux_contrib_density_energy%z + flux_contrib_i_density_energy%z)
            flux_i_momentum%x = flux_i_momentum%x + factor * &
                 (ff_flux_contrib_momentum_x%z + flux_contrib_i_momentum_x%z)
            flux_i_momentum%y = flux_i_momentum%y + factor * &
                 (ff_flux_contrib_momentum_y%z + flux_contrib_i_momentum_y%z)
            flux_i_momentum%z = flux_i_momentum%z + factor * &
                 (ff_flux_contrib_momentum_z%z + flux_contrib_i_momentum_z%z)
          end if
        end do

        fluxes(i + 1 + VAR_DENSITY * nelr) = flux_i_density
        fluxes(i + 1 + (VAR_MOMENTUM + 0) * nelr) = flux_i_momentum%x
        fluxes(i + 1 + (VAR_MOMENTUM + 1) * nelr) = flux_i_momentum%y
        fluxes(i + 1 + (VAR_MOMENTUM + 2) * nelr) = flux_i_momentum%z
        fluxes(i + 1 + VAR_DENSITY_ENERGY * nelr) = flux_i_density_energy
      end do
    end do
  end subroutine compute_flux

  subroutine time_step(j, nelr, old_variables, variables, step_factors, fluxes)
    integer, intent(in) :: j
    real, allocatable, intent(in) :: old_variables(:), step_factors(:), fluxes(:)
    real, allocatable, intent(out) :: variables(:)
    integer :: blk, b_start, b_end, i, i1
    integer :: m0, m1, m2, d, de
    real :: factor

    do concurrent (blk : 1, nelr / BLOCK_LENGTH)
      b_start = (blk - 1) * BLOCK_LENGTH
      if (blk * BLOCK_LENGTH > nelr) then
        b_end = nelr
      else
        _end = blk * BLOCK_LENGTH
      end if

      do i = b_start, b_end - 1
        d = VAR_DENSITY * nelr
        de = VAR_DENSITY_ENERGY * nelr
        m0 = (VAR_MOMENTUM + 0) * nelr
        m1 = (VAR_MOMENTUM + 1) * nelr
        m2 = (VAR_MOMENTUM + 2) * nelr
        i1 = i + 1

        factor = step_factors(i1) / real(RK + 1 - j)
        variables(i1 + d) = old_variables(i1 + d) + factor * fluxes(i1 + d)
        variables(i1 + m0) = old_variables(i1 + m0) + factor * fluxes(i1 + m0)
        variables(i1 + m1) = old_variables(i1 + m1) + factor * fluxes(i1 + m1)
        variables(i1 + m2) = old_variables(i1 + m2) + factor * fluxes(i1 + m2)
        variables(i1 + de) = old_variables(i1 + de) + factor * fluxes(i1 + de)
      end do
    end do
  end subroutine time_step

end module euler3d

program main
  use euler3d

  implicit none

  integer :: argc, iterations, rate
  character(256) :: data_file_name
  integer :: i, j, k, idx
  integer :: nel, nelr
  real, allocatable :: areas, normals,
  integer, allocatable :: elements_surrounding_elements
  integer :: start_total_time, end_total_time, start_time, end_time
  integer :: copy_start, copy_end, sf_start, sf_end, rk_start, rk_end
  real(8) :: elapsed_time, total_time, time
  real(8) :: rk_mean, rk_std_dev, rk_sum
  real :: ff_pressure, ff_speed_of_sound, ff_speed, angle_of_attack
  type(Float3) :: velocity, momentum
  type(Float3) :: ff_flux_contrib_momentum_x, ff_flux_contrib_momentum_y
  type(Float3) :: ff_flux_contrib_momentum_z, ff_flux_contrib_density_energy

  argc = command_argument_count()
  if (argc < 2) then
    write (*,*) "Specify data file name"
    stop 1
  end if

  iterations = 4000
  if (argc > 2) then
    iterations = command_argument_to_i(2)
  end if

  call get_command_argument(1, data_file_name)

  write (*, "(/A/)") "---- euler3d benchmark (forall) ----"
  write (*, "(A,A)") "  Input file : ", data_file_name
  write (*, "(A,A,A/)") "  Iterations: ", iterations, "."

  write (*, "(A)", advance="no") &
       "  Reading input data, allocating arrays, initializing data ..."

  call system_clock(RATE=rate)
  call system_clock(total_start_time)

  ! These need to be computed the first time in order to compute the time step.
  allocate(ff_variable(NVAR))

  angle_of_attack = (3.1415926535897931 / 180.0) * DEG_ANGLE_OF_ATTACK
  ff_variable(VAR_DENSITY + 1) = 1.4

  ff_pressure = 1.0
  ff_speed_of_sound = sqrt(GAMMA * ff_pressure / ff_variable(VAR_DENSITY + 1))
  ff_speed = FF_MACH * ff_speed_of_sound

  ff_velocity%x = ff_speed * cos(angle_of_attack)
  ff_velocity%y = ff_speed * sin(angle_of_attack)
  ff_velocity%z = 0.0

  ff_variable(VAR_MOMENTUM + 1) = ff_variable(VAR_DENSITY + 1) * ff_velocity%x
  ff_variable(VAR_MOMENTUM + 2) = ff_variable(VAR_DENSITY + 1) * ff_velocity%y
  ff_variable(VAR_MOMENTUM + 3) = ff_variable(VAR_DENSITY + 1) * ff_velocity%z
  ff_variable(VAR_DENSITY_ENERGY + 1) = ff_variable(VAR_DENSITY + 1) * &
        (0.5 * (ff_speed ** 2)) + (ff_pressure / (GAMMA - 1.0))

  ff_momentum%x = ff_variable(VAR_MOMENTUM + 1)
  ff_momentum%y = ff_variable(VAR_MOMENTUM + 2)
  ff_momentum%z = ff_variable(VAR_MOMENTUM + 3)

  call compute_flux_contribution(ff_variable[VAR_DENSITY], ff_momentum, &
       ff_variable[VAR_DENSITY_ENERGY + 1], ff_pressure, ff_velocity, &
       ff_flux_contrib_momentum_x, ff_flux_contrib_momentum_y, &
       ff_flux_contrib_momentum_z, ff_flux_contrib_density_energy)

  open (unit=10, file=data_file_name, action="read")
  read (10, *) nel
  nelr = block_length * ((nel / block_length ) + min(1, rem(nel, block_length)))

  allocate(areas(nelr))
  allocate(elements_surrounding_elements(nelr * NNB))
  allocate(normals(NDIM * NNB * nelr))

  do i = 1, nel
    read (10, *) areas(i)
    do j = 1, NNB
      read (10, *) elements_surrounding_elements(i + (j - 1) * nelr)
      if (elements_surrounding_elements(i + (j - 1) * nelr) < 0) then
        elements_surrounding_elements(i + (j - 1) * nelr) = -1
      end if

      do k = 1, NDIM
        idx = i + (j - 1 + (k - 1) * NNB) * nelr
        read (10, *) normals(idx)
        normals(idx) = -normals(idx)
      end do
    end do
  end do

  close(10)

  ! Fill in remaining data
  last = nel
  do i = nel + 1, nelr
    areas(i) = areas(last)
    do j = 1, NNB
      ! duplicate the last element
      elements_surrounding_elements(i + (j - 1) * nelr) = &
           elements_surrounding_elements(last + (j - 1) * nelr)
      do k = 1, NDIM
        normals(i + (j + (k - 1) * NNB) * nelr) = &
             normals(last + (j - 1 + (k - 1) * NNB) * nelr)
      end do
    end do
  end do

  allocate(variables(nelr * NVAR))
  allocate(old_variables(nelr * NVAR))
  allocate(fluxes(nelr * NVAR))
  allocate(step_factors(nelr))
  allocate(rk_times(iterations))

  write (*,"(A/)") " done"

  write (*, "(A)") "  Starting benchmark ... "
  call system_clock(start_time)
  call initialize_variables(nelr, variables, ff_variable)

  ! Begin iterations
  copy_total = 0.0
  sf_total = 0.0
  rk_total = 0.0
  do i = 1, iterations
    call system_clock(copy_start)
    call cpy(old_variables, variables, nelr * NVAR)
    call system_clock(copy_end)
    time = dble(copy_end - copy_start) / rate
    copy_total = copy_total + time

    call system_clock(sf_start)
    call compute_step_factor(nelr, variables, areas, step_factors)
    call system_clock(sf_end)
    time = dble(sf_end - sf_start) / rate
    sf_total = sf_total + time

    call system_clock(rk_start)
    do j = 1, RK
      call compute_flux(nelr, elements_surrounding_elements, normals, &
           variables, fluxes, ff_variable, ff_flux_contrib_momentum_x, &
           ff_flux_contrib_momentum_y, ff_flux_contrib_momentum_z, &
           ff_flux_contrib_density_energy)
      call time_step(j, nelr, old_variables, variables, step_factors, fluxes)
    end do
    call system_clock(rk_end)
    time = dble(rk_end - rk_start) / rate
    if (i > 1) then
      rk_times(i) = time
      rk_total = rk_total + time
    end do
  end do

  dump_variables(variables, nel, nelr)

  call system_clock(total_end_time)
  elapsed_time = (total_end_time - start_time) / rate
  total_time = (end_time - total_start_time) / rate
  rk_mean = rk_total / (iterations - 1)
  sum = 0.0
  do i = 2, iterations
    dist = rk_times(i) - rk_mean
    sum = sum + (dist ** 2)
  end do
  rk_std_dev = sqrt(sum / (iterations - 1))

  write(*,*)
  write(*,"(A,F10.3,A)") "      Total time : ", total_time, " seconds."
  write(*,"(A,F10.3,A)") "    Compute time : ", elapsed_time, " seconds."
  write(*,"(A,F10.3,A,F10.3,A)") "            copy : ", copy_total, &
       " seconds (average: ", copy_total / iterations, " seconds)."
  write(*,"(A,F10.3,A,F10.3,A)") "              sf : ", sf_total, &
       " seconds (average: ", sf_total / iterations, " seconds)."
  write(*,"(A,F10.3,A,F10.3,A,F10.3)") "              rk : ", rk_total, &
       " seconds (average: ", rk_mean, " seconds / std dev: ", rk_std_dev, ")."

  deallocate(ff_variable)
  deallocate(areas)
  deallocate(elements_surrounding_elements)
  deallocate(normals)
  deallocate(variables)
  deallocate(old_variables)

contains

  integer function command_argument_to_i(n)
    implicit none

    integer, intent(in) :: n
    character(32) :: buf

    call get_command_argument(n, buf)
    read (buf,*) command_argument_to_i
  end function command_argument_to_i

  integer pure function rem(m, n)
    implicit none
    integer, intent(in) :: m, n
    if (m < n) then
       rem = m
    else
       rem = mod(m, n)
    end if
  end function rem

  subroutine dump(variables, nel, nelr)
    real, allocatable, intent(in) :: variables(:)
    integer, intent(in) :: nel, nelr
    integer, parameter :: unit = 10
    integer :: i, j

    open(unit=unit, file="density-fort.dat", action="write")
    write(unit, "(I0,A,I0)") nel, " ", nelr
    do i = 1, nel
      write(unit, "(F10.8,A)") variables(i + VAR_DENSITY * nelr)
    end do
    close(unit)

    open(unit=unit, file="momentum-fort.dat", action="write")
    write(unit, "(I0,A,I0)") nel, " ", nelr
    do i = 1, nel
      do j = 1, NDIM
        write(unit, "(F10.8,A)", advance="no") &
             variables(i + (VAR_MOMENTUM + j - 1) * nelr), " "
      end do
      write(unit, *) ""
    end do
    close(unit)

    open(unit=unit, file="density_energy-fort.dat", action="write")
    write(unit, "(I0,A,I0)") nel, " ", nelr
    do i = 1, nel
      write(unit, "(F10.8)") variables(i + VAR_DENSITY_ENERGY * nelr)
    end do
    close(unit)

  end subroutine dump

end program main
