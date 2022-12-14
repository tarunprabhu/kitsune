include ../experiments.mk

experiment = euler3d
exp_forall = ${experiment}-forall
exp_kokkos = ${experiment}-kokkos

cxx_flags += -I${kitsune_prefix}/include

ifeq ($(KITSUNE_ENABLE_CUDA),TRUE)
  targets += ${exp_forall}.cuda.${host_arch}
  targets += ${exp_kokkos}.cuda.${host_arch}
endif

ifeq ($(KITSUNE_ENABLE_HIP), TRUE)
  targets += ${exp_kokkos}.hip.${host_arch}
endif

VPATH := ./forall/:./kokkos/

all: ${targets}

forall_lto_src = forall/compute_flux.cpp \
   forall/compute_step_factor.cpp \
   forall/time_step.cpp

forall_src = forall/euler3d-forall.cpp

forall_lto_objs := $(patsubst %.cpp, forall/%.lto.o, $(notdir $(forall_lto_src)))
forall_objs := $(patsubst %.cpp, forall/%.o, $(notdir $(forall_src)))
forall/%.lto.o: forall/%.cpp
	$(KITSUNE_TIME_COMPILES) ${clangxx} -c -flto ${cxx_flags} ${kitflags} ${tapir_cu_flags} -o $@ $<
forall/%.o: forall/%.cpp
	$(KITSUNE_TIME_COMPILES) ${clangxx} -c ${cxx_flags} ${kitflags} ${tapir_cu_flags} -o $@ $<

${exp_forall}.cuda.${host_arch}:  ${forall_lto_objs} ${forall_objs}
	$(KITSUNE_TIME_COMPILES) ${clangxx} -flto -fuse-ld=lld -ftapir=cuda \
	${tapir_cu_lto_flags} -o $@ ${forall_lto_objs} ${forall_objs} ${link_flags} \
	-Xlinker -rpath=${kitsune_prefix}/lib

###########################################
#
# Generate LLVM IR and linked LLVM bitcode. Useful for debugging.
#
forall_ir_objs := $(patsubst %.cpp, %.ll, $(forall_lto_src))

forall/%.ll: forall/%.cpp
	${clangxx} -S -emit-llvm ${cxx_flags} ${kitflags} ${tapir_cu_flags} -o $@ $<

${exp_forall}.cuda.${host_arch}.bc: ${forall_ir_objs}
	${clangxx} -flto -fuse-ld=gold -Wl,-plugin-opt=emit-llvm  -ftapir=cuda \
	-o $@ ${forall_ir_objs} ${link_flags}

${exp_forall}.cuda.${host_arch}.ll: ${exp_forall}.cuda.${host_arch}.bc
	${llvm_dis} -o $@ $<
#
###########################################

kokkos_src = kokkos/compute_flux.cpp \
   kokkos/compute_step_factor.cpp \
   kokkos/time_step.cpp \
   kokkos/euler3d-kokkos.cpp
kokkos_objs := $(patsubst %.cpp, kokkos/%.o, $(notdir $(kokkos_src)))
kokkos/%.o: kokkos/%.cpp
	$(KITSUNE_TIME_COMPILES) ${clangxx} -c ${cxx_flags} ${clang_cu_flags} ${kokkos_cu_flags} \
		-I${kitsune_prefix}/include -o $@ $<

${exp_kokkos}.cuda.${host_arch}: ${kokkos_objs}
	$(KITSUNE_TIME_COMPILES) ${clangxx} ${link_flags} -o $@ ${kokkos_objs} ${kokkos_ld_flags} ${kokkos_ld_libs} \
		-Xlinker -rpath=${kokkos_cu_prefix}/lib

clean:
	-rm -f *.${host_arch} *.o forall/*.o  kokkos/*.o
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin
	-rm -f *.ppm *.jpg
	-rm -f *.ll *.ptx *.csv *.log *.s *.fbin *.tapir
	-rm -f *.dat
