
set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  IRReader
  JITLink
  Object
  OrcDebugging
  OrcJIT
  OrcShared
  OrcTargetProcess
  Passes
  RuntimeDyld
  Support
  TargetParser
  native
  )

add_llvm_unittest(OrcJITTests
  CoreAPIsTest.cpp
  ExecutorAddressTest.cpp
  ExecutionSessionWrapperFunctionCallsTest.cpp
  EPCGenericJITLinkMemoryManagerTest.cpp
  EPCGenericMemoryAccessTest.cpp
  IndirectionUtilsTest.cpp
  JITTargetMachineBuilderTest.cpp
  LazyCallThroughAndReexportsTest.cpp
  LookupAndRecordAddrsTest.cpp
  MachOPlatformTest.cpp
  MapperJITLinkMemoryManagerTest.cpp
  MemoryFlagsTest.cpp
  MemoryMapperTest.cpp
  ObjectFormatsTest.cpp
  ObjectLinkingLayerTest.cpp
  OrcCAPITest.cpp
  OrcTestCommon.cpp
  ResourceTrackerTest.cpp
  RTDyldObjectLinkingLayerTest.cpp
  SharedMemoryMapperTest.cpp
  SimpleExecutorMemoryManagerTest.cpp
  SimplePackedSerializationTest.cpp
  SymbolStringPoolTest.cpp
  TaskDispatchTest.cpp
  ThreadSafeModuleTest.cpp
  WrapperFunctionUtilsTest.cpp
  JITLinkRedirectionManagerTest.cpp
  ReOptimizeLayerTest.cpp

  EXPORT_SYMBOLS
  )

target_link_libraries(OrcJITTests PRIVATE
                        LLVMTestingSupport
                        ${ORC_JIT_TEST_LIBS})

# It is not clear why just this test has issues with linking. This became a
# problem once lib/Kitsune was moved to kitsune/lib. This serves as a workaround
# for now, but it would be good to figure out why we had to do this.
if (NOT LLVM_LINK_LLVM_DYLIB)
  target_link_libraries(OrcJITTests PRIVATE
    $<LINK_LIBRARY:WHOLE_ARCHIVE,LLVMAnalysis,LLVMProfileData,LLVMObject>)
endif ()
