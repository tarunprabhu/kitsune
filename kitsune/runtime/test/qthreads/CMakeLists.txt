set(TESTS
  Launch.cpp
  WorkerID.cpp
)

# These tests call functions from Qthreads directly, and require appropriate
# include and link options.
set(TESTS_WITH_QTHREADS
  NumThreads.cpp
)

# These tests call PAPI's API functions directly, and require appropriate
# include and link options.
set(TESTS_WITH_PAPI
  PAPIInit.cpp
)

add_kitrt_systests("${TESTS}")

add_kitrt_systests("${TESTS_WITH_QTHREADS}"
  INCDIRS ${KITSUNE_QTHREADS_INCDIRS}
  LIBS ${KITSUNE_QTHREADS_LIBS})

if (KITSUNE_PAPI_ENABLED)
  add_kitrt_systests("${TESTS_WITH_PAPI}"
    INCDIRS "${KITSUNE_PAPI_INCDIRS}"
    LIBS "${KITSUNE_PAPI_LIBS}")
endif ()
