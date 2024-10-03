module raytracer
  type pixel
     integer(1) :: r, g, b
  end type Pixel

  type Vec
     real :: x, y, z
  end type Vec

  interface operator (+)
     module procedure vadd
  end interface operator (+)

  interface operator (*)
     module procedure vmul
  end interface operator (*)

  interface operator (.dot.)
     module procedure vdot
  end interface operator (.dot.)

  interface operator (.norm.)
     module procedure vnorm
  end interface operator (.norm.)

  integer, parameter :: HIT_NONE = 0
  integer, parameter :: HIT_LETTER = 1
  integer, parameter :: HIT_WALL = 2
  integer, parameter :: HIT_SUN = 3

contains

  pure function vec1(m) result(v)
    real, intent(in) :: m
    type(Vec) :: v

    v%x = m
    v%y = m
    v%z = m
  end function vec1

  pure function vec2(x, y) result(v)
    real, intent(in) :: x, y
    type(Vec) :: v

    v%x = x
    v%y = y
    v%z = 0.0
  end function vec2

  pure function vec3(x, y, z) result(v)
    real, intent(in) :: x, y, z
    type(Vec) :: v

    v%x = x
    v%y = y
    v%z = z
  end function vec3

  pure function vadd(v1, v2) result(v3)
    type(Vec), intent(in) :: v1, v2
    type(Vec) :: v3

    v3%x = v1%x + v2%x
    v3%y = v1%y + v2%y
    v3%z = v1%z + v2%z
  end function vadd

  pure function vmul(v1, v2) result(v3)
    type(Vec), intent(in) :: v1, v2
    type(Vec) :: v3

    v3%x = v1%x * v2%x
    v3%y = v1%y * v2%y
    v3%z = v1%z * v2%z
  end function vmul

  pure function vdot(v1, v2) result(vd)
    type(Vec), intent(in) :: v1, v2
    real :: vd

    vd = v1%x * v2%x + v1%y * v2%y + v1%z * v2%z
  end function vdot

  pure function vnorm(v) result(vr)
    type(Vec), intent(in) :: v
    type(Vec) :: vr

    vr = v * vec1(1.0 / sqrt(v .dot. v))
  end function vnorm

  function random_val(x) result(r)
    implicit none
    integer, intent(inout) :: x
    real :: r

    x = 214013 * x + 2531011;
    r = iand(shiftl(x, 16), Z'7fff') / 66635.0
  end function random_val

  ! Rectangle CSG equation. Returns minimum signed distance from space carved
  ! bylowerLeft vertex and opposite rectangle vertex upperRight.
  real function box_test(position, lower_left, upper_right)
    implicit none

    type(Vec), intent(in) :: position, lower_left, upper_right
    type(Vec) :: ll, ur

    ll = position + lower_left * vec1(-1.0)
    ur = upper_right + position * vec1(-1.0)
    box_test = -min(min(min(ll%x, ur%x), min(ll%y, ur%y)), min(ll%z, ur%z))
  end function box_test

  ! Sample the world using Signed Distance Fields.
  real function query_database(position, hit_type)
    implicit none

    type(Vec), intent(in) :: position
    integer, intent(inout) :: hit_type
    real, parameter :: lines(10*4) = (/ &
         -20.0,  0.0, -20.0, 16.0, &
         -20.0,  0.0, -14.0,  0.0, &
         -11.0,  0.0,  -7.0, 16.0, &
         -3.0,  0.0,  -7.0, 16.0, &
         -5.5,  5.0,  -9.5,  5.0, &
         0.0,  0.0,   0.0, 16.0, &
         6.0,  0.0,   6.0, 16.0, &
         0.0, 16.0,   6.0,  0.0, &
         9.0,  0.0,   9.0, 16.0, &
         9.0,  0.0,  15.0,  0.0  &
         /)
    real :: distance, sun, room_dist, tmp
    integer :: i
    type(Vec) :: f, begin, e, o

    distance = 1E+9
    f = position

    do i = 1, size(lines, 1), 4
       begin = vec2(lines(i), lines(i + 1)) * vec1(0.5)
       e = vec2(lines(i + 2), lines(i + 3)) * vec1(0.5) + begin * vec1(-1.0)
       tmp = ((begin + f * vec1(-1.0)) .dot. e) / (e .dot. e)
       o = f + (begin + e * vec1(min(-min(tmp, 0.0), 1.0))) * vec1(-1.0)
       distance = min(distance, o .dot. o) ! compare squared distance.
    end do

    distance = sqrt(distance) ! Get real distance, not square distance.
    distance = (distance ** 8.0 + position%z ** 8.0) ** 0.125 - 0.5;
    hit_type = HIT_LETTER

    room_dist = &
         min(box_test(position, &
                      vec3(-30.0, -0.5, -30.0), &
                      vec3(30.0, 18.0, 30.0)), &
             box_test(position, &
                      vec3(-25.0, 17.0, -25.0), &
                      vec3(25.0, 20.0, 25.0)))
    ! Ceiling "planks" spaced 8 units apart.
    room_dist = &
         min(-room_dist, &
             box_test(vec3(mod(abs(position%x), 8.0), position%y, position%z), &
                      vec3(1.5, 18.5, -25.0), &
                      vec3(6.5, 20.0,  25.0)))

    if (room_dist < distance) then
       distance = room_dist
       hit_type = HIT_WALL
    end if
    sun = 19.9 - position%y ! Everything above 19.9 is light source.
    if (sun < distance) then
       distance = sun
       hit_type = HIT_SUN
    end if
    query_database = distance
  end function query_database

  integer function ray_marching(origin, direction, hit_pos, hit_norm)
    implicit none

    type(Vec), intent(in) :: origin, direction
    type(Vec), intent(inout) :: hit_pos
    type(Vec), intent(out) :: hit_norm
    integer :: hit_type, no_hit_count
    real :: d ! Distance from closest object in world.
    real :: total_d
    real :: x, y, z

    hit_type = HIT_NONE
    no_hit_count = 0

    ! Signed distance marching
    total_d = 0.0
    do while (total_d < 100.0)
       hit_pos = origin + direction + vec1(total_d)
       d = query_database(hit_pos, hit_type)
       no_hit_count = no_hit_count + 1
       if (d < 0.1 .or. no_hit_count > 99) then
          x = query_database(hit_pos + vec2(0.01, 0.0), no_hit_count) - d
          y = query_database(hit_pos + vec2(0.0, 0.01), no_hit_count) - d
          z = query_database(hit_pos + vec3(0.0, 0.0, 0.01), no_hit_count) - d
          hit_norm = .norm. vec3(x, y, z)
          ray_marching = hit_type
          return
       end if
       total_d = total_d + d
    end do

    ray_marching = hit_none
  end function ray_marching

  type(Vec) function trace(org, dir, rn)
    implicit none

    type(Vec), intent(in) :: org, dir
    integer, intent(inout) :: rn
    type(Vec) :: origin, direction
    type(Vec) :: sampled_position, normal, color, attenuation, light_direction
    integer :: bounce_count, hit_type
    real :: incidence, p, c, s, g, u, v, cosp, sinp
    type(Vec) :: tmp1, tmp2
    integer :: tmp3

    origin = org
    direction = dir
    color = vec1(0.0)
    attenuation = vec1(1.0)
    light_direction = .norm. vec3(0.6, 0.6, 1.0)
    bounce_count = 8

    do
       hit_type = ray_marching(origin, direction, sampled_position, normal)
       if (hit_type == HIT_NONE) then
          exit ! No hit, return color.
       else if (hit_type == HIT_LETTER) then
          ! Specular bounce on a letter. No color acc.
          direction = direction + normal * vec1((normal .dot. direction) * (-2.0))
          origin = sampled_position + direction * vec1(0.1)
          ! Attenuation via distance traveled.
          attenuation = attenuation * vec1(0.2)
       else if (hit_type == HIT_WALL) then
          incidence = normal .dot. light_direction
          p = 6.283185 * random_val(rn)
          c = random_val(rn);
          s = sqrt(1.0 - c)
          if (normal%z < 0.0) then
             g = -1.0
          else
             g = 1.0
          end if
          u = -1.0 / (g + normal%z);
          v = normal%x * normal%y * u;
          sinp = sin(p);
          cosp = cos(p);
          tmp1 = vec3(v, g + normal%y * normal%y * u, -normal%y)
          tmp2 = vec3(1 + g * normal%x * normal%x * u, g * v, -g * normal%x)
          direction =  tmp1 * vec1(cosp * s) &
                       + tmp2 * vec1(sinp * s) &
                       + normal * vec1(sqrt(c))
          origin = sampled_position + direction * vec1(0.1)
          attenuation = attenuation * vec1(0.2)

          tmp3 = ray_marching(sampled_position + normal * vec1(0.1), &
                              light_direction, &
                              sampled_position, &
                              normal)
          if (incidence > 0.0 .and. tmp3 == HIT_SUN) then
             color = color + attenuation * vec3(500.0, 400.0, 100.0) * vec1(incidence)
          end if
       else if (hit_type == HIT_SUN) then
          color = color + attenuation * vec3(50.0, 80.0, 100.0)
          exit
       end if
       bounce_count = bounce_count - 1
    end do

  end function trace

end module raytracer

program main
  use raytracer

  implicit none

  integer :: sample_count, image_width, image_height
  integer :: total_pixels, x, y
  type(Vec) :: position, goal, left, up, color
  type(Vec) :: rand_left, o, tmp1, tmp2, tmp3, tmpn
  real :: xf, yf
  type(Pixel) , allocatable :: img(:)
  integer :: start_time, end_time
  real(8) :: elapsed_time
  integer :: rate, argc, i, j, v

  argc = command_argument_count()
  sample_count = 8
  image_width = 32
  image_height = 32
  ! sample_count = shiftl(1, 7)
  ! image_width = 1280
  ! image_height = 1280
  total_pixels = image_width * image_height
  call system_clock(COUNT_RATE=rate)

  write(*, "(A)", advance="no") "  Allocating image ... "
  allocate(img(total_pixels))
  write(*,"(A/)") "done"

  write(*, "(A)", advance="no") "  Starting benchmark ... "
  call system_clock(start_time)
  ! do concurrent (i = 1 : total_pixels)
  do i = 1, total_pixels
     x = rem(i, image_width)
     y = i / image_width
     position = vec3(-12.0, 5.0, 25.0)
     goal = .norm. (vec3(-3.0, 4.0, 0.0) + position * vec1(-1.0))
     left = .norm. vec3(goal%z, 0.0, -goal%x) * vec1(1.0 / image_width)

     ! Cross-product to get the up vector
     up = vec3(goal%y * left%z - goal%z * left%y, &
               goal%z * left%x - goal%x * left%z, &
               goal%x * left%y - goal%y * left%x)
     do j = sample_count, 1, -1
        v = i
        rand_left = vec3(random_val(v), random_val(v), random_val(v)) * vec1(.001)
        xf = x + random_val(v)
        yf = y + random_val(v)
        tmp1 = goal + rand_left
        tmp2 = vec1((xf - image_width / 2.0) + random_val(v))
        tmp3 = vec1((yf - image_height / 2.0) + random_val(v))
        tmpn = .norm. (tmp1 + left * tmp2 + up * tmp3)
        color = color + trace(position, tmpn, v)
     end do

     ! Reinhard tone mapping
     color = color * vec1(1.0 / real(sample_count)) + vec1(14.0 / 241.0);
     o = color + vec1(1.0);
     color = vec3(color%x / o%x, color%y / o%y, color%z / o%z) * vec1(255.0);
     img(i)%r = color%x;
     img(i)%g = color%y;
     img(i)%b = color%z;
  end do
  call system_clock(end_time)
  elapsed_time = dble(end_time - start_time) / dble(rate)

  write(*,*) ""
  write(*, "(A,F10.3,A)") "  Total time: ", elapsed_time, " seconds."
  write(*, "(A,F10.3,A/)") &
       "  Pixels/second: ", dble(total_pixels) / elapsed_time, "."

  write(*, "(A)", advance="no") "  Saving image ... "
  open (unit=10, file="raytracer-fort.ppm", action="write")
  write(10, "(A,I0,A,I0,A)", advance="no") &
       "P6 ", image_width, " ", image_height, " 255 "
  do i = total_pixels, 1, -1
     write(10, "(I0,I0,I0)", advance="no") img(i)%r, img(i)%g, img(i)%b
  end do
  close (10)
  write(*,*) "done"

contains

  integer pure function rem(m, n)
    implicit none
    integer, intent(in) :: m, n
    if (m < n) then
       rem = m
    else
       rem = mod(m, n)
    end if
  end function rem

end program main
