set(LLVM_LINK_COMPONENTS
  KitFrontend
  Core
)

add_kitsune_unittest_with_input_files(KitFrontendTests
  ReductionUtilsTest.cpp

  EXPORT_SYMBOLS
)

target_include_directories(KitFrontendTests PUBLIC
  ${KITSUNE_SOURCE_DIR}/lib)

target_link_libraries(KitFrontendTests PRIVATE LLVMTestingSupport)
