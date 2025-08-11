set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  Core
  KitCore
  KitSupport
  Support
)

add_llvm_tool(kit-mdc
  kit-mdc.cpp)
