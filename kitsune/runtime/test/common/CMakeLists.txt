set(TESTS
  logEarly.cpp
  logError.cpp
  logFatal.cpp
  logNormal.cpp
  logWarning.cpp
  timerCannotWriteToFile.c
  timerDefaultStderr.c
  timerEmpty.c
  timerIterateSingleThread.c
  timerIterateThreads.c
  timerSingleThread.c
  timerThreads.c
  timerWriteToFile.c
  timerWriteNothingToFile.c
)

add_kitrt_systests("${TESTS}")
