set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  CodeGen
  Core
  KitCore
  KitLinker
  KitObject
  KitSupport
  Passes
  Support
  TargetParsers
)

add_llvm_tool(kit-linkdc
  kit-linkdc.cpp)
