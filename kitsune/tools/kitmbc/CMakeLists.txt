set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  BitWriter
  Core
  IRReader
  KitCore
  KitSupport
  Support
)

add_llvm_tool(kitmbc
  kitmbc.cpp)
