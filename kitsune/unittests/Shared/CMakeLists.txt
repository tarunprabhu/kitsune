set(LLVM_LINK_COMPONENTS
  KitCore
  KitSupport
  Support
)

add_kitsune_unittest(KitShared
  TypeTraitsTest.cpp
)
