set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  CodeGen
  Core
  IRReader
  KitCore
  KitSupport
  Support
)

add_llvm_tool(kitmenc
  kitmenc.cpp)
