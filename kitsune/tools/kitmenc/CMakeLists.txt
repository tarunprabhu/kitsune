set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  CodeGen
  Core
  IRReader
  KitCore
  KitSupport
  Support
)

add_llvm_tool(kitmenc
  kitmenc.cpp)
