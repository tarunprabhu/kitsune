# This module will set the following variables in your project:
#
#  Realm_FOUND       -- Realm was found.
#  Realm_INCLUDE_DIR -- directory with Realm header files.
#  Realm_LIBRARY     -- full path to the Realm library.
#  Realm_LIBRARY_DIR -- path to where the Realm library is installed.
#  Realm_LINK_LIBS   -- set of link libraries (e.g. -lrealm)

find_path(Realm_INCLUDE_DIR realm.h
  PATHS /usr/local/include
        /opt/include
        /opt/local/include
        $ENV{REALM_PATH}/include
)

find_library(Realm_LIBRARY realm
  PATHS /usr/local/lib64
        /usr/local/lib
	      /opt/lib64
	      /opt/lib
	      /opt/local/lib64
	      /opt/local/lib
        $ENV{REALM_PATH}/lib
        $ENV{REALM_PATH}/lib64
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Realm DEFAULT_MSG
  Realm_INCLUDE_DIR
	Realm_LIBRARY
)

if (Realm_FOUND)

  get_filename_component(Realm_LIBRARY_DIR
                         ${Realm_LIBRARY}
                         DIRECTORY
                         CACHE)

  # Basic set of libraries -- will potenitally need additional
  # libraries based on Realm is built.
  set(Realm_LINK_LIBS "-lrealm -lpthread -ldl -lrt"
    CACHE
    STRING "List of libraries needed for Realm linking.")
endif()

