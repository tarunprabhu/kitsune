set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  Analysis
  AsmParser
  BitWriter
  Core
  KitAnalysis
  KitCore
)

add_kitsune_unittest_with_input_files(KitAnalysisTests
  TTObjectsAnalysisTest.cpp

  EXPORT_SYMBOLS
)

target_link_libraries(KitAnalysisTests PRIVATE LLVMTestingSupport)

# On AIX, enable run-time linking to allow symbols from the plugins shared
# objects to be properly bound.
if(CMAKE_SYSTEM_NAME STREQUAL "AIX")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-brtl")
endif()
