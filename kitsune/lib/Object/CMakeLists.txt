add_llvm_component_library(LLVMKitObject
  ArchiveUtils.cpp
  BinaryUtils.cpp
  EmbDeviceCode.cpp
  EmbDeviceCodeContext.cpp
  ObjectUtils.cpp

  COMPONENT_NAME
  KitObject

  LINK_COMPONENTS
  CodeGen
  Core
  MC
  Object
  Passes
  Support
  TargetParser
)
