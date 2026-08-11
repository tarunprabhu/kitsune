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
)

add_kitrt_systests("${TESTS}")
