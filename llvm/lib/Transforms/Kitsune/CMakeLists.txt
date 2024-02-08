add_llvm_component_library(LLVMKitsuneOpts
  LowerMobileIntrinsics.cpp
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
