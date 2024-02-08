include ../experiments.mk 

cxx_flags += -I${kitsune_prefix}/include  -fPIC

targets = math-calls.opencilk.${host_arch}

ifeq ($(KITSUNE_ENABLE_CUDA), TRUE)
  targets += math-calls.cuda.${host_arch}
  targets += math-calls-kokkos.nvcc.${host_arch}
  targets += math-calls-kokkos.clang.${host_arch}
endif 

ifeq ($(KITSUNE_ENABLE_HIP), TRUE)
  targets += math-calls.hip.${host_arch}
endif 

all: ${targets} 

#############################
# Kitsune+Tapir versions.
#
math-calls.opencilk.${host_arch}: math-calls.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_opencilk_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
math-calls.cuda.${host_arch}: math-calls.cpp
	/usr/bin/time ${clangxx} ${cxx_flags} ${kitflags} ${tapir_cu_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
math-calls.hip.${host_arch}: math-calls.cpp
	/usr/bin/time ${clangxx} -mllvm -debug-only=hipabi ${cxx_flags} ${kitflags} ${tapir_hip_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
#
#############################

clean:
	-rm -f *.${host_arch}
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin *.o *.a 
	-rm -f *.ppm *.jpg
	-rm -f *.ll *.ptx *.csv *.log *.s *.fbin *.tapir

