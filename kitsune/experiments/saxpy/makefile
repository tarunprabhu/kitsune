include ../common.mk
include ../cuda.mk
include ../kokkos.mk
include ../gcc.mk 

# General (shared) flags (see common.mk)
clang_flags=${cxx_flags} ${opt_flags} ${clang_info_flags}
# kitsune kokkos mode flags.
kit_kokkos_flags=-fkokkos -fkokkos-no-init


#### Tapir parameters 
COARSEN_FACTOR?=1
GRAINSIZE?=1
THREADS_PER_BLOCK?=256
BLOCKS_PER_GRID?=0

#### Kitsune+Tapir options 
#  Notes:
#   -ftapir=cuda will target the CUDA ABI transform. 
#   -cuabi-verbose dumps kernel stats via ptxas output. 
#   -cuabi-keep-files saves intermediate files (.ll, .ptx, .fatbin) on disk for a closer look at codegen details. 
#
tapir_cu_launch_flags=-I${CUDA_HOME}/include
tapir_cu_flags=-mllvm --debug-only=cuda-abi -ftapir=cuda -ffp-contract=fast -mllvm -cuabi-keep-files -mllvm -cuabi-verbose -mllvm -cuabi-arch=${NVARCH} ${tapir_cu_launch_flags}
tapir_stripmine_flags=-mllvm -stripmine-count=${GRAINSIZE} -mllvm -stripmine-coarsen-factor=${COARSEN_FACTOR} -mllvm -cuabi-default-grainsize=${GRAINSIZE}
####

#### CUDA flags
nvcc_flags=--std c++17 -m64 -arch=${NVARCH} --resource-usage -O3 --no-exceptions -I${kitsune_prefix}/include
clang_cuda_flags=-xcuda --cuda-gpu-arch=${NVARCH} ${cxx_flags} ${opt_flags} -I${kitsune_prefix}/include
####

#### Kokkos stuff 
kokkos_cu_flags=-I${kokkos_cu_prefix}/include ${nvcc_cxx_flags} -m64 -arch=${NVARCH} --resource-usage -O3 --no-exceptions -I${kitsune_prefix}/include


all: saxpy-forall.cuda.${host_arch}
saxpy-forall.cuda.${host_arch}: saxpy-forall.cpp
	${clangxx} ${clang_flags} ${tapir_cu_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib

clean:
	-rm -f *.${host_arch}
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin

realclean:
	-rm -f *.ll *.ptx *.csv *.log
