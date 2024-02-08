add_llvm_component_library(LLVMFrontendTapir
  CommandLine.cpp
  OptLevelUtils.cpp
  Tapir.cpp
  TapirTargetOptions.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend/Tapir

  COMPONENT_NAME
  FrontendTapir

  LINK_COMPONENTS
  Core
  Support
  CodeGen
)
