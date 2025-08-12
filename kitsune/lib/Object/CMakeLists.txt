add_llvm_component_library(LLVMKitObject
  ArchiveUtils.cpp
  BinaryUtils.cpp
  # EmbDeviceCode.cpp
  ObjectUtils.cpp

  # FIXME: Remove this once we have sorted out the testing
  PARTIAL_SOURCES_INTENDED

  COMPONENT_NAME
  KitObject

  LINK_COMPONENTS
  Object
  Support
)
