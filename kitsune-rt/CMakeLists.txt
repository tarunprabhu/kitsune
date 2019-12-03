
find_path(REALM_INCLUDE_DIR  realm.h)
find_library(REALM_LIBRARY realm)
get_filename_component(REALM_LIBRARY_DIR
		       ${REALM_LIBRARY}
		       DIRECTORY
		       CACHE)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Realm DEFAULT_MSG
				  REALM_INCLUDE_DIR 
				  REALM_LIBRARY)

include_directories(
	${REALM_INCLUDE_DIR}
	)
link_directories(
	${REALM_LIBRARY_DIR}	
	)

file(GLOB realm_srcs Realm/*.cc)

add_library(kitsunerealm SHARED ${realm_srcs} )

target_link_libraries(kitsunerealm PUBLIC realm)
