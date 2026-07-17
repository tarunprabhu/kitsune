set(TESTS
  EnvNumThreads.c
  EnvVerbose.c
  LogEarly.cpp
  LogEarlyNoTag.cpp
  LogError.cpp
  LogErrorNoTag.cpp
  LogFatal.cpp
  LogFatalNoTag.cpp
  LogNormal.cpp
  LogNormalNoTag.cpp
  LogWarning.cpp
  LogWarningNoTag.cpp
  TimerCannotWriteToFile.c
  TimerDefaultStderr.c
  TimerEmpty.c
  TimerIterateSingleThread.c
  TimerIterateThreads.c
  TimerSingleThread.c
  TimerThreads.c
  TimerWriteToFile.c
  TimerWriteNothingToFile.c
)

# These tests call PAPI's API functions directly, and require appropriate
# include and link options.
set(TESTS_WITH_PAPI
  PAPIMultipleFini.c
  PAPIMultipleInit.c)

add_kitrt_systests("${TESTS}")

if (KITSUNE_PAPI_ENABLED)
  add_kitrt_systests("${TESTS_WITH_PAPI}"
    INCDIRS "${KITSUNE_PAPI_INCDIRS}"
    LIBS "${KITSUNE_PAPI_LIBS}")
endif ()
