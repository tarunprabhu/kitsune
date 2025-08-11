add_llvm_component_library(LLVMKitObject
  EmbDeviceCode.cpp
  EmbDeviceCodeParser.cpp

  COMPONENT_NAME
  KitObject

  LINK_COMPONENTS
  Object
  Support
)
