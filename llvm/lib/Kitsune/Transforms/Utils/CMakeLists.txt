add_llvm_component_library(LLVMKitTransformUtils
  EmbModulePassUtils.cpp

  DEPENDS

  LINK_COMPONENTS
  Core
  IRReader
  KitCore
  KitSupport
  Linker
  Support
)
