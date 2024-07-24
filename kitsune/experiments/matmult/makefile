include ../experiments.mk

ifeq ($(BUILD_CUDA_EXPERIMENTS),true)
  targets += matmul-forall.cuda.${host_arch}
endif

all: ${targets}

# forall-based tests
matmul-forall.cuda.${host_arch}: matmul-forall.cpp
	@echo $@
	@$(TIME_CMD) $(KIT_CXX) $(TAPIR_CUDA_FLAGS) -o $@ $< -Xlinker -rpath=$(KITSUNE_PREFIX)/lib
	@$(FILE_SIZE)

#####
define newline 


endef 

define RUN_test
	$$(./$(1) > $(1).log) $(newline) 
endef

run: $(targets) 
	@echo "running generated executables..."
	$(foreach prog,$(targets),$(call RUN_test,$(prog)))

launch_bench: matmul-forall.cuda.${host_arch}
	../launch-bench.sh $<
#####

.PHONY: all clean run

clean:
	-rm -f *.${host_arch} *.ll *.o
	-rm -f *~ core *~ *.log 

