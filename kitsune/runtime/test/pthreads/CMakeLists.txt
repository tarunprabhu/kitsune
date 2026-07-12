set(TESTS
  NumThreads.c
  RTInitFini.c)

set(TESTS_WITH_PAPI
  PAPIInitFini.c)

add_kitrt_systests(${TESTS})

add_kitrt_systest(${TESTS_WITH_PAPI}
  INCDIRS ${KITSUNE_PAPI_INCDIRS}
  LIBS ${KITSUNE_PAPI_LIBS})
