set(TESTS
  RTInitFini.c
)

# These tests call PAPI's API functions directly, and require appropriate
# include and link options.
set(TESTS_WITH_PAPI
  PAPIInitFini.c
)

# These tests call functions from Qthreads directly, and require appropriate
# include and link options.
set(TESTS_WITH_CUDA
  Initialized.cpp
  MultipleFini.cpp
  MultipleInit.cpp
)

add_kitrt_systests("${TESTS}")

add_kitrt_systests("${TESTS_WITH_CUDA}"
  INCDIRS ${KITSUNE_CUDA_INCDIRS})

if (KITSUNE_PAPI_ENABLED)
  add_kitrt_systests("${TESTS_WITH_PAPI}"
    INCDIRS "${KITSUNE_PAPI_INCDIRS}"
    LIBS "${KITSUNE_PAPI_LIBS}")
endif ()
