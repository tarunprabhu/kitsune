set(TESTS
  Memset.cpp
)

# These tests call PAPI's API functions directly, and require appropriate
# include and link options.
set(TESTS_WITH_PAPI
  PAPIInit.cpp
)

# We may include kitcuda.h, which currently itself includes the hip headers,
# so we have to provide a path where they can be found.
add_kitrt_systests("${TESTS}"
  INCDIRS ${KITSUNE_HIP_INCDIRS})

if (KITSUNE_PAPI_ENABLED)
  add_kitrt_systests("${TESTS_WITH_PAPI}"
    INCDIRS "${KITSUNE_PAPI_INCDIRS}"
    LIBS "${KITSUNE_PAPI_LIBS}")
endif ()
