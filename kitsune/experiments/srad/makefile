include ../experiments.mk

experiment = srad
exp_forall = ${experiment}-forall
exp_kokkos = ${experiment}-kokkos

cxx_flags += -I${kitsune_prefix}/include

targets = ${exp_forall}.opencilk.${host_arch}

ifeq ($(KITSUNE_ENABLE_CUDA),TRUE)
  targets += ${exp_forall}.cuda.${host_arch}
  targets += ${exp_kokkos}.nvcc.${host_arch}
  targets += ${exp_kokkos}.clang.${host_arch}
endif
ifeq ($(KITSUNE_ENABLE_HIP),TRUE)
  targets += ${exp_forall}.hip.${host_arch}
endif

all: ${targets}

#############################
# Kitsune+Tapir versions.
${exp_forall}.opencilk.${host_arch}: ${exp_forall}.cpp
	$(KITSUNE_TIME_COMPILES) ${clangxx} ${cxx_flags} ${kitflags} ${tapir_opencilk_flags} -o $@ $< \
	-Xlinker -rpath=${kitsune_prefix}/lib
${exp_forall}.cuda.${host_arch}: ${exp_forall}.cpp
	$(KITSUNE_TIME_COMPILES) ${clangxx} ${cxx_flags} ${kitflags} ${tapir_cu_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
${exp_forall}.hip.${host_arch}: ${exp_forall}.cpp
	$(KITSUNE_TIME_COMPILES) ${clangxx} ${cxx_flags} ${kitflags} ${tapir_hip_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib
#############################


#############################
# Kitsune+Tapir versions that only dump LLVM-IR. Intended for debugging.
#
${exp_forall}.opencilk.${host_arch}.ll: ${exp_forall}.cpp
	${clangxx} -S -emit-llvm ${cxx_flags} ${kitflags} ${tapir_opencilk_flags} -o $@ $<
${exp_forall}.cuda.${host_arch}.ll: ${exp_forall}.cpp
	${clangxx} -S -emit-llvm ${cxx_flags} ${kitflags} ${tapir_cu_flags} -o $@ $<
${exp_forall}.hip.${host_arch}.ll: ${exp_forall}.cpp
	${clangxx} -S -emit-llvm ${cxx_flags} ${kitflags} ${tapir_hip_flags} -o $@ $<
${exp_forall}.kokkos.${host_arch}.ll: ${exp_kokkos}.cpp
	${clangxx} -S -emit-llvm ${cxx_flags} ${kitflags} ${tapir_cu_flags} ${kokkos_cu_flags} ${kitsune_kokkos_flags} -o $@ $<
#
#############################


#############################
# Kokkos version w/ both nvcc and clang for comparison.
${exp_kokkos}.nvcc.${host_arch}: ${exp_kokkos}.cpp
	$(KITSUNE_TIME_COMPILES) ${nvcc_wrapper} ${kokkos_nvcc_flags} -I${kitsune_prefix}/include -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib
${exp_kokkos}.clang.${host_arch}: ${exp_kokkos}.cpp
	$(KITSUNE_TIME_COMPILES) ${clangxx} ${cxx_flags} ${clang_cu_flags} ${kokkos_cu_flags} -I${kitsune_prefix}/include -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib


clean:
	-rm -f *.${host_arch}
	-rm -f *.fbin *.s *.ptx *.ll
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin
	-rm -f *.csv *.log *.s
