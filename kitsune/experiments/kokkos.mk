
# Assumes we've enabled Kokkos in kitsune build (which will install
# the serial version into the same prefix as the kitsune/llvm toolchain.
kokkos_prefix=${kitsune_prefix}

# Architecture/platform-specific installs of Kokkos.
kokkos_cu_prefix=${kitsune_prefix}/opt/kokkos_cuda
nvcc_wrapper=${kokkos_prefix}/bin/nvcc_wrapper

kokkos_cxx_flags=-I${kokkos_prefix}/include
kokkos_ld_flags=-L${kokkos_prefix}/lib
kokkos_ld_libs=-lkokkoscore -lcuda -ldl -lrt
