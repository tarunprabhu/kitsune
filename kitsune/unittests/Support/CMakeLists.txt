set(LLVM_LINK_COMPONENTS
  KitCore
  KitSupport
  Support
)

add_kitsune_unittest(KitSupport
  MaybeBoolTest.cpp
  ToStringTest.cpp
  TypeTraitsTest.cpp
)
