# These paths are relative to each experiment's directory... 
# Note: Be cautious with ordering here -- there are some 
# dependencies... 
include ../inc/config.mk
include ../inc/common.mk

include ../inc/clang.mk
include ../inc/cuda.mk
include ../inc/hip.mk
include ../inc/kokkos.mk

include ../inc/kitsune-tapir.mk

