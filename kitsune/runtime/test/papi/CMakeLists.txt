set(TESTS
  Basic.cpp
  CannotWriteToFile.cpp
  Convenience.cpp
  DefaultEvents.cpp
  DefaultStderr.cpp
  Empty.cpp
  IterateSingleThread.cpp
  IterateThreads.cpp
  Nested.cpp
  SingleThread.cpp
  Threads.cpp
  Unknown.cpp
  WriteNothingToFile.cpp
  WriteToFile.cpp
)

add_kitrt_systests("${TESTS}"
  INCDIRS "${KITSUNE_PAPI_INCDIRS}"
  LIBS "${KITSUNE_PAPI_LIBS}")
