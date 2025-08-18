add_llvm_component_library(LLVMKitLinker
  DeviceCodeLinker.cpp

  LINK_COMPONENTS
  CodeGen
  Core
  KitCore
  KitObject
  KitSupport
  Passes
  MC
  Support
  TargetParser
)
