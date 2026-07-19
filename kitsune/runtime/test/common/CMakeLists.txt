set(TESTS
  EnvNumThreads.cpp
  EnvVerbose.c

  LogError.cpp
  LogErrorNoTag.cpp
  LogFatal.cpp
  LogFatalNoTag.cpp
  LogIfVerbose.cpp
  LogIfVerboseNoTag.cpp
  LogNormal.cpp
  LogNormalNoTag.cpp
  LogWarning.cpp
  LogWarningNoTag.cpp

  RTInitialized.c
  RTMultipleInit.c
  RTMultipleFini.c

  TimerBasic.c
  TimerCannotWriteToFile.c
  TimerDefaultStderr.c
  TimerEmpty.c
  TimerInitFini.c
  TimerInitialized.c
  TimerIterateSingleThread.c
  TimerIterateThreads.c
  TimerMultipleFini.c
  TimerMultipleInit.c
  TimerSingleThread.c
  TimerThreads.c
  TimerWriteToFile.c
  TimerWriteNothingToFile.c
)

# These tests call PAPI's API functions directly, and require appropriate
# include and link options.
set(TESTS_WITH_PAPI
  RTPAPIInitFini.c
  PAPIMultipleFini.c
  PAPIMultipleInit.c
)

add_kitrt_systests("${TESTS}")

if (KITSUNE_PAPI_ENABLED)
  add_kitrt_systests("${TESTS_WITH_PAPI}"
    INCDIRS "${KITSUNE_PAPI_INCDIRS}"
    LIBS "${KITSUNE_PAPI_LIBS}")
endif ()
