set(TESTS
  Initialized.c
  MultipleInit.c
  MultipleFini.c
  NumPartials.c
  NumThreads.c
  RTInitFini.c)

# These tests call PAPI's API functions directly, and require appropriate
# include and link options.
set(TESTS_WITH_PAPI
  PAPIInitFini.c)

add_kitrt_systests("${TESTS}")

add_kitrt_systest("${TESTS_WITH_PAPI}"
  INCDIRS "${KITSUNE_PAPI_INCDIRS}"
  LIBS "${KITSUNE_PAPI_LIBS}")
