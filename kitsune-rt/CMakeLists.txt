
include_directories(
	${REALM_INCLUDE_DIR}
	)
link_directories(
	${REALM_LIBRARY_DIR}	
	)

file(GLOB realm_srcs Realm/*.cc)

add_library(kitsunerealm SHARED ${realm_srcs} )

target_link_libraries(kitsunerealm PUBLIC realm)
