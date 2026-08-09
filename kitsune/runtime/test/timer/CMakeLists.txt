set(TESTS
  Basic.cpp
  CannotWriteToFile.cpp
  DefaultStderr.cpp
  Empty.cpp
  InitFini.cpp
  IterateSingleThread.cpp
  IterateThreads.cpp
  SingleThread.cpp
  Threads.cpp
  WriteToFile.cpp
  WriteNothingToFile.cpp
)

add_kitrt_systests("${TESTS}")
