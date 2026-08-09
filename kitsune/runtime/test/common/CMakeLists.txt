set(TESTS
  EnvNumThreads.cpp
  EnvVerbose.cpp

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

  RTInitialized.cpp
  RTMultipleInit.cpp
  RTMultipleFini.cpp

  TimerBasic.cpp
  TimerCannotWriteToFile.cpp
  TimerDefaultStderr.cpp
  TimerEmpty.cpp
  TimerInitFini.cpp
  TimerIterateSingleThread.cpp
  TimerIterateThreads.cpp
  TimerSingleThread.cpp
  TimerThreads.cpp
  TimerWriteToFile.cpp
  TimerWriteNothingToFile.cpp
)

# These tests call PAPI's API functions directly, and require appropriate
# include and link options.
set(TESTS_WITH_PAPI
  PAPIBasic.cpp
  PAPICannotWriteToFile.cpp
  PAPIConvenience.cpp
  PAPIDefaultEvents.cpp
  PAPIDefaultStderr.cpp
  PAPIEmpty.cpp
  PAPIIterateSingleThread.cpp
  PAPIIterateThreads.cpp
  PAPINested.cpp
  PAPISingleThread.cpp
  PAPIThreads.cpp
  PAPIUnknown.cpp
  PAPIWriteNothingToFile.cpp
  PAPIWriteToFile.cpp
)

add_kitrt_systests("${TESTS}")

if (KITSUNE_PAPI_ENABLED)
  add_kitrt_systests("${TESTS_WITH_PAPI}"
    INCDIRS "${KITSUNE_PAPI_INCDIRS}"
    LIBS "${KITSUNE_PAPI_LIBS}")
endif ()
