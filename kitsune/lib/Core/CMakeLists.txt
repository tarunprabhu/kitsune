add_llvm_component_library(LLVMKitCore
  AddrSpaceUtils.cpp
  ArgAttrs.cpp
  ArgUtils.cpp
  AttrsImpl.cpp
  BasicBlockUtils.cpp
  ConstantUtils.cpp
  Diagnostics.cpp
  DIUtils.cpp
  EmbUtils.cpp
  FuncAttrs.cpp
  FuncUtils.cpp
  GVAttrs.cpp
  GVUtils.cpp
  InstAttrs.cpp
  InstUtils.cpp
  IntrinsicUtils.cpp
  KernelProperties.cpp
  LoopAttrs.cpp
  LoopUtils.cpp
  MetadataUtils.cpp
  ModuleAttrs.cpp
  ModuleUtils.cpp
  TapirLoopUtils.cpp
  TTOptions.cpp
  TTPlugin.cpp
  TTUtils.cpp
  TargetUtils.cpp
  TypeUtils.cpp
  ValueUtils.cpp
  Verifier.cpp

  DEPENDS
  intrinsics_gen
  kitsune_generated

  # There is a circular dependence here between this library and LLVMCore. For
  # more details, see llvm/lib/IR/CMakeLists.txt. It would be nice if we could
  # get rid of these, but that is highly unlikely.
  LINK_COMPONENTS
  Analysis
  BitReader
  BitWriter
  CodeGen
  Core
  KitSupport
  MC
  Support
  TapirOpts
  TargetParser
)
