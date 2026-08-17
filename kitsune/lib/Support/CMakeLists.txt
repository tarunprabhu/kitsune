# This library *MUST NOT* depend on LLVMCore. It is ok to depend on LLVMSupport
# and any other libraries that LLVMSupport depends on. But it must never depend
# on any libraries that transitively depend on LLVMCore.

add_llvm_component_library(LLVMKitSupport
  CommandLineOptions.cpp
  CommandLineUtils.cpp
  ErrorHandling.cpp
  ErrorUtils.cpp
  MaybeBool.cpp
  ToString.cpp

  COMPONENT_NAME
  KitSupport

  DEPENDS
  kitsune_generated

  LINK_COMPONENTS
  Support
)
