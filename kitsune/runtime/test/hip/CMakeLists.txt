set(TESTS
  Initialized.cpp
  MultipleFini.cpp
  MultipleInit.cpp
  RTInitFini.cpp
)

# These tests call PAPI's API functions directly, and require appropriate
# include and link options.
set(TESTS_WITH_PAPI
  PAPIInitFini.cpp
)

add_kitrt_systests("${TESTS}")

if (KITSUNE_PAPI_ENABLED)
  add_kitrt_systests("${TESTS_WITH_PAPI}"
    INCDIRS "${KITSUNE_PAPI_INCDIRS}"
    LIBS "${KITSUNE_PAPI_LIBS}")
endif ()
