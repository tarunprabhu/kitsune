set(TESTS
  Initialized.c
  Launch.c
  MultipleInit.c
  MultipleFini.c
  NumPartials.c
  RTInitFini.c
  WorkerID.c)

# These tests call PAPI's API functions directly, and require appropriate
# include and link options.
set(TESTS_WITH_PAPI
  PAPIInitFini.c)

# These tests call functions from Qthreads directly, and require appropriate
# include and link options.
set(TESTS_WITH_QTHREADS
  NumThreads.c)

add_kitrt_systests("${TESTS}")

if (KITSNUE_PAPI_ENABLED)
  add_kitrt_systest("${TESTS_WITH_PAPI}"
    INCDIRS ${KITSUNE_PAPI_INCDIRS}
    LIBS ${KITSUNE_PAPI_LIBS})
endif ()

add_kitrt_systests("${TESTS_WITH_QTHREADS}"
  INCDIRS ${KITSUNE_QTHREADS_INCDIRS}
  LIBS ${KITSUNE_QTHREADS_LIBS})
