add_llvm_component_library(LLVMKitCore
  AddrSpaceUtils.cpp
  CommandLineOptions.cpp
  ConstantUtils.cpp
  EmbUtils.cpp
  FunctionUtils.cpp
  GlobalVariableUtils.cpp
  IntrinsicUtils.cpp
  KernelProperties.cpp
  LoopAttrs.cpp
  LoopUtils.cpp
  MetadataUtils.cpp
  ModuleUtils.cpp
  TTOptions.cpp
  TTPlugin.cpp
  TTUtils.cpp
  TargetUtils.cpp
  TypeUtils.cpp

  DEPENDS
  intrinsics_gen
  kitsune_generated_attrs

  # There is a circular dependence here between this library and LLVMCore. For
  # more details, see llvm/lib/IR/CMakeLists.txt. It would be nice if we could
  # get rid of these, but that is highly unlikely.
  LINK_COMPONENTS
  BitReader
  BitWriter
  CodeGen
  Core
  KitSupport
  MC
  Support
  TargetParser
)
