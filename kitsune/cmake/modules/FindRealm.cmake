# 
#  Realm_FOUND       -- realm was found.
#  Realm_INCLUDE_DIR -- directory with realm header files. 
#  Realm_LIBRARY     -- full path to the realm library. 
#  Realm_LIBRARY_DIR -- path to where the realm library is installed. 
#  Realm_WRAPPER_LIBRARY_DIR -- path to where the kitsune-rt realm wrapper library is installed. 
#  Realm_LINK_LIBS   -- set of link libraries (e.g. -lrealm) 
# 

message(STATUS "kitsune: looking for realm...")

find_path(Realm_INCLUDE_DIR  realm.h)
find_library(Realm_LIBRARY realm)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Realm DEFAULT_MSG
                                  Realm_INCLUDE_DIR 
                                  Realm_LIBRARY)

if (Realm_FOUND) 
  message(STATUS "kitsune: looking for realm... FOUND")
  get_filename_component(Realm_LIBRARY_DIR
                         ${Realm_LIBRARY}
                         DIRECTORY
                         CACHE)

  # Basic set of libraries -- will potenitally need additional
  # libraries based on Realm is built.
  set(Realm_LINK_LIBS "-lrealm -lpthread -ldl -lrt"
    CACHE
    STRING "List of libraries needed for Realm linking.")

  message(STATUS "Realm runtime include directory: ${Realm_INCLUDE_DIR}")
  message(STATUS "Realm runtime library directory: ${Realm_LIBRARY_DIR}")
  message(STATUS "Realm runtime link libraries   : ${Realm_LINK_LIBS}")

else()
  message(STATUS "kitsune: looking for realm... NOT FOUND")
  set(KITSUNE_ENABLE_REALM FALSE CACHE BOOL "Enable automatic include and library flags for Realm.")
  set(Realm_LINK_LIBS "" CACHE STRING "List of libraries need to link with for Realm.")
endif()

