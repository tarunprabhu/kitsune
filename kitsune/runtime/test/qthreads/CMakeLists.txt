function(add_systest source)
  add_kitrt_systest(${source}
    INCDIRS ${KITSUNE_QTHREADS_INCDIRS}
    LIBS ${KITSUNE_QTHREADS_LIBS})
endfunction ()

add_systest(qthrNumThreadsTest.c)
