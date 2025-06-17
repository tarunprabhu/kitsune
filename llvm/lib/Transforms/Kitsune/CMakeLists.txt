add_llvm_component_library(LLVMKitsuneOpts
  EmbBCPassUtils.cpp
  FinalizeKernelMetadata.cpp
  GenerateKitsuneCtors.cpp
  GenerateKitsuneCtorCuda.cpp
  GenerateKitsuneCtorHip.cpp
  LinkDeviceBitcode.cpp
  LowerKitsuneRuntimeIntrinsics.cpp
  LowerMobileIntrinsics.cpp
  OptimizeEmbBC.cpp
  PrepareEmbBC.cpp
  ResolveDeviceFuncs.cpp
  StripKitsuneAddrSpace.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Kitsune

  DEPENDS
  intrinsics_gen
  LLVMLinker
  LLVMBitWriter

  COMPONENT_NAME
  KitsuneOpts

  LINK_COMPONENTS
  Analysis
  Core
  IRReader
  Linker
  MC
  Scalar
  Support
  TransformUtils
  Passes
)

target_include_directories(LLVMKitsuneOpts PRIVATE
  ${CMAKE_SOURCE_DIR}/../kitsune/include)
