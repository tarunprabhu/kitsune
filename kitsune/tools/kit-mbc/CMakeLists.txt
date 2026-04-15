set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  BitWriter
  CodeGen
  Core
  IRReader
  KitCore
  KitSupport
  Support
)

add_llvm_tool(kit-mbc
  kit-mbc.cpp

  DEPENDS
  intrinsics_gen
  kitsune_generated)
