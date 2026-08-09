set(TESTS
)

# These tests call functions from ROCm directly, and require appropriate include
# and link options.
set(TESTS_WITH_HIP
)

# These tests call PAPI's API functions directly, and require appropriate
# include and link options.
set(TESTS_WITH_PAPI
  PAPIInit.cpp
)

add_kitrt_systests("${TESTS}")

add_kitrt_systests("${TESTS_WITH_HIP}"
  INCDIRS ${KITSUNE_HIP_INCDIRS})

if (KITSUNE_PAPI_ENABLED)
  add_kitrt_systests("${TESTS_WITH_PAPI}"
    INCDIRS "${KITSUNE_PAPI_INCDIRS}"
    LIBS "${KITSUNE_PAPI_LIBS}")
endif ()
