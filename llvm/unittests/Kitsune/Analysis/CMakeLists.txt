set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  BitWriter
  Core
  KitAnalysis
  KitCore
)

add_llvm_unittest_with_input_files(KitAnalysisTests
  TapirTargetAnalysisTest.cpp

  EXPORT_SYMBOLS
)

add_dependencies(KitAnalysisTests intrinsics_gen)

target_link_libraries(KitAnalysisTests PRIVATE LLVMTestingSupport)

# On AIX, enable run-time linking to allow symbols from the plugins shared
# objects to be properly bound.
if(CMAKE_SYSTEM_NAME STREQUAL "AIX")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-brtl")
endif()
