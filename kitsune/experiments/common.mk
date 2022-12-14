SHELL=/bin/bash
host_arch := $(shell uname -o -m | awk '{print $$1}')
$(info kitsune experiments host architecture: ${host_arch})

ifneq ($(KITSUNE_OPT_LEVEL),)
  opt_level = $(KITSUNE_OPT_LEVEL)
else
  opt_level = 3
endif

ifneq ($(KITSUNE_OPT_FLAGS),)
  opt_flags = $(KITSUNE_OPT_FLAGS)
else
  opt_flags = -march=native -O${opt_level}
endif

# KITSUNE_TIME_COMPILES should contain the command to time the compile and any
# flags the command takes. The command is usually just 'time' but it could be
# '/usr/bin/time -v' or something similar if something more accurate/capable is
# needed. The default is not to time anything.
ifneq ($(KITSUNE_TIME_COMPILES),)
  time_compile_cmd = $(KITSUNE_TIME_COMPILES)
else
  time_compile_cmd =
endif

##################################
# EXPERIMENTS-WIDE C and CXX FLAGS
#c_flags = -march=native $(CFLAGS)
c_flags = $(CFLAGS)
cxx_flags = -std=c++17 -fno-exceptions $(CXX_FLAGS)

ifneq ($(KITSUNE_VERBOSE),)
  clang_flags = ${clang_flags} -v
  clangxx_flags = ${clangxx_flags} -v
endif

$(info common c flags: ${c_flags})
$(info common c++ flags: ${cxx_flags})
