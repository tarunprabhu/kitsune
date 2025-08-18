set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  CodeGen
  Core
  KitCore
  # KitLinker [FIXME: Enable this once it is available]
  KitObject
  KitSupport
  Passes
  Support
  TargetParsers
)

add_llvm_tool(kit-linkdc
  kit-linkdc.cpp)
