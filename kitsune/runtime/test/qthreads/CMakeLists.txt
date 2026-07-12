set(TESTS
  RTInitFini.c)

set(TESTS_WITH_PAPI
  PAPIInitFini.c)

# These are tests that call functions from Qthreads directly, and must be
# provided with paths to the qthreads header and libraries.
set(TESTS_WITH_QTHREADS
  NumThreads.c)

add_kitrt_systests(${TESTS})

add_kitrt_systest(${TESTS_WITH_PAPI}
  INCDIRS ${KITSUNE_PAPI_INCDIRS}
  LIBS ${KITSUNE_PAPI_LIBS})

add_kitrt_systests(${TESTS_WITH_QTHREADS}
  INCDIRS ${KITSUNE_QTHREADS_INCDIRS}
  LIBS ${KITSUNE_QTHREADS_LIBS})
