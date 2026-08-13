set(TESTS
  DisabledCuda.c
  DisabledHip.c
  DisabledOpenCilk.c
  DisabledPAPI.c
  DisabledQthreads.c
  InitFini.c
  Initialized.cpp
  MultipleInit.cpp
  MultipleFini.cpp
  RefCounts.c
)

add_kitrt_systests("${TESTS}")
