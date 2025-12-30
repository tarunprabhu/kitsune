add_custom_target(kitsune-doc)
set_target_properties(kitsune-doc PROPERTIES FOLDER "Kitsune/Docs")

# This is a convenience synonym for kitsune-doc. The latter is present for
# consistency with other LLVM subprojects.
add_custom_target(kitsune-docs)
add_dependencies(kitsune-docs kitsune-doc)

function (gen_attrs_rst rst td target)
  get_filename_component(TABLEGEN_INCLUDE_DIR ${td} DIRECTORY)
  list(APPEND LLVM_TABLEGEN_FLAGS "-I${TABLEGEN_INCLUDE_DIR}")
  list(APPEND LLVM_TABLEGEN_FLAGS
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../clang/include/")
  clang_tablegen(${rst} -gen-kitsune-attr-docs
    SOURCE ${td}
    TARGET ${target})
endfunction()

function (gen_options_rst rst td target)
  get_filename_component(TABLEGEN_INCLUDE_DIR ${td} DIRECTORY)
  list(APPEND LLVM_TABLEGEN_FLAGS "-I${TABLEGEN_INCLUDE_DIR}")
  list(APPEND LLVM_TABLEGEN_FLAGS
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../clang/include/clang/Driver/")
  clang_tablegen(${rst} -gen-kitsune-opt-docs
    SOURCE ${td}
    TARGET ${target})
endfunction()

if (KITSUNE_ENABLE_DOXYGEN)
  find_package(Doxygen)
  find_program(dot_path dot REQUIRED)

  # First, set a number of variables that are required before configuring
  # doxygen.cfg.in.
  set(abs_srcdir ${CMAKE_CURRENT_SOURCE_DIR})
  set(abs_builddir ${CMAKE_CURRENT_BINARY_DIR})
  set(tapir_incdir ${CMAKE_SOURCE_DIR}/include/llvm/Transforms/Tapir)
  if (LLVM_DOXYGEN_EXTERNAL_SEARCH)
    set(enable_searchengine "YES")
    set(searchengine_url "${LLVM_DOXYGEN_SEARCHENGINE_URL}")
    set(enable_server_based_search "YES")
    set(enable_external_search "YES")
    set(extra_search_mappings "${LLVM_DOXYGEN_SEARCH_MAPPINGS}")
  else()
    set(enable_searchengine "NO")
    set(searchengine_url "")
    set(enable_server_based_search "NO")
    set(enable_external_search "NO")
    set(extra_search_mappings "")
  endif()

  if (KITSUNE_ENABLE_LLVM_DOXYGEN)
    # The top-level CMakeLists.txt file is llvm/CMakeLists.txt. As a result,
    # CMAKE_SOURCE_DIR is llvm/.
    set(llvm_incdir ${CMAKE_SOURCE_DIR}/include)
    set(llvm_libdir ${CMAKE_SOURCE_DIR}/lib)
  endif ()

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg @ONLY)

  # Now that doxygen.cfg.in has been configured, unset the variables, just in
  # case.
  set(dot_path)
  set(abs_srcdir)
  set(abs_builddir)
  set(enable_searchengine)
  set(searchengine_url)
  set(enable_server_based_search)
  set(enable_external_search)
  set(extra_search_mappings)

  # Enable some additional debugging options when building LLVM's API
  # documentation so we have some idea of progress being made since building
  # LLVM's API documentation is time-consuming.
  if (KITSUNE_ENABLE_LLVM_DOXYGEN)
    set(DOXYGEN_DEBUG_FLAGS -dsections)
  endif ()
  add_custom_target(doxygen-kitsune
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_DEBUG_FLAGS} ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating kitsune doxygen documentation."
    VERBATIM USES_TERMINAL)
  set_target_properties(doxygen-kitsune PROPERTIES FOLDER "Kitsune/Docs")

  add_dependencies(kitsune-docs doxygen-kitsune)
  if (LLVM_BUILD_DOCS)
   add_dependencies(doxygen doxygen-kitsune)
  endif()

  # Create an alias for doxygen-kitsune to be consistent with the other
  # documentation-related targets specific to Kitsune
  add_custom_target(kitsune-docs-doxygen)
  add_dependencies(kitsune-docs-doxygen doxygen-kitsune)

  # We install the API documentation together with the rest of the HTML
  # documentation. This is not what the other LLVM subprojects do. However,
  # self-contained documentation is more important for us than compatibility
  # with LLVM in this case.
  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html/
      DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/kitsune/www/doxygen)
  endif()
endif ()

if (KITSUNE_ENABLE_SPHINX)
  set(HTML_DIR ${CMAKE_CURRENT_BINARY_DIR}/SourceHtml)
  set(MAN_DIR ${CMAKE_CURRENT_BINARY_DIR}/SourceMan)

  set(KITSUNE_REPO_HOST "github.com" CACHE INTERNAL
    "URL where the official Kitsune repo is hosted" FORCE)
  set(KITSUNE_REPO_OWNER "lanl" CACHE INTERNAL
    "Username of the owner of the official Kitsune repo" FORCE)

  # Right now, we do not use the main branch on the Kitsune official repo. We
  # will at some point, but for now, it is best to point users to point to one
  # of the dev/* branches since they are likely to be more "stable" than the
  # other branches
  set(KITSUNE_REPO_BRANCH "dev/${LLVM_VERSION_MAJOR}.x" CACHE INTERNAL
    "The current branch in Kitsune's repo" FORCE)

  # Since the top-level CMakeLists.txt file is in the llvm/ directory of this
  # checkout, CMAKE_SOURCE_DIR will point to that directory. The true root is
  # one level up from there.
  get_filename_component(CMAKE_SOURCE_ROOT_DIR "${CMAKE_SOURCE_DIR}" DIRECTORY)

  # The AddSphinxTarget module only looks for the sphinx executable if
  # LLVM_ENABLE_SPHINX=ON. Since we may only be interested in building Kitsune's
  # documentation, not all of LLVM's, we do not set LLVM_ENABLE_SPHINX=ON even
  # if KITSUNE_ENABLE_SPHINX=ON. However, the user may well have set both if
  # they were interested in building all documentation. Therefore, save the
  # value of LLVM_ENABLE_SPHINX, and "temporarily" set it to ON just so we
  # can find sphinx.
  set(LLVM_ENABLE_SPHINX_BAK ${LLVM_ENABLE_SPHINX})
  set(LLVM_ENABLE_SPHINX ON)
  include(AddSphinxTarget)
  set(LLVM_ENABLE_SPHINX ${LLVM_ENABLE_SPHINX_BAK})

  if (SPHINX_FOUND)
    set(ENV_VARS
      CMAKE_SOURCE_DIR="${CMAKE_SOURCE_ROOT_DIR}"
      KITSUNE_REPO_HOST="${KITSUNE_REPO_HOST}"
      KITSUNE_REPO_OWNER="${KITSUNE_REPO_OWNER}"
      KITSUNE_REPO_BRANCH="${KITSUNE_REPO_BRANCH}"
      KITSUNE_DEFAULT_LANGS="${KITSUNE_DEFAULT_LANGS}"
      KITSUNE_DEFAULT_TAPIR_TARGETS="${KITSUNE_DEFAULT_TAPIR_TARGETS}"
      KITSUNE_GUARANTEED_TAPIR_TARGETS="${KITSUNE_GUARANTEED_TAPIR_TARGETS}"
      KITSUNE_KNOWN_TAPIR_TARGETS="${KITSUNE_KNOWN_TAPIR_TARGETS}"
      KITSUNE_CUDA_ARCH_DEFAULT="${KITSUNE_CUDA_ARCH_DEFAULT}"
      KITSUNE_CUDA_VERSION_MIN="${KITSUNE_CUDA_VERSION_MIN}"
      KITSUNE_CUDA_VERSION_MAX="${KITSUNE_CUDA_VERSION_MAX}"
      KITSUNE_HIP_ARCH_DEFAULT="${KITSUNE_HIP_ARCH_DEFAULT}"
      KITSUNE_HIP_VERSION_MIN="${KITSUNE_HIP_VERSION_MIN}"
      KITSUNE_HIP_VERSION_MAX="${KITSUNE_HIP_VERSION_MAX}")

    # CLANG_TABLEGEN_EXE must be set for clang_tablegen to run without errors
    find_program(CLANG_TABLEGEN_EXE "clang-tblgen"
      PATHS ${LLVM_TOOLS_BINARY_DIR}
      NO_DEFAULT_PATH)

    # Create the output directories
    add_custom_target(kitsune-html-dir
      COMMAND ${CMAKE_COMMAND} -E make_directory ${HTML_DIR})
    add_custom_target(kitsune-man-dir
      COMMAND ${CMAKE_COMMAND} -E make_directory ${MAN_DIR})

    if (SPHINX_OUTPUT_HTML)
      # The environment variables here are read directly in conf.py.
      # sphinx-build accepts a SPHINXOPT environment variable, but it seems as
      # if any -D options in that variable are not recognized.
      add_sphinx_target(html kitsune SOURCE_DIR "${HTML_DIR}"
        ENV_VARS "${ENV_VARS}")

      gen_attrs_rst(${HTML_DIR}/KitAttrDoc.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitAttrDoc.td
        kit-attrs-doc)

      gen_options_rst(${HTML_DIR}/KitClangOptionsDoc.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitClangOptionsDoc.td
        kit-clang-options-doc)
      add_dependencies(kit-clang-options-doc kitsune-html-dir)

      gen_options_rst(${HTML_DIR}/KitFlangOptionsDoc.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitFlangOptionsDoc.td
        kit-flang-options-doc)
      add_dependencies(kit-flang-options-doc kitsune-html-dir)

      # Sphinx needs all the input files in one place for it to work. Since
      # some of the input files, such as the command-line option references,
      # are generated, they will be in the build directory. This means that the
      # static (non-generated) files also need to be copied to the build
      # directory before invoking sphinx-build.
      add_custom_target(copy-html-inputs
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/conf.py ${HTML_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/*.md ${HTML_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/CommandGuide ${HTML_DIR}/CommandGuide
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/doxygen ${HTML_DIR}/doxygen
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/_themes/kitsune-theme
        ${HTML_DIR}/_themes/kitsune-theme
        DEPENDS kitsune-html-dir)

      add_dependencies(docs-kitsune-html
        kit-attrs-doc
        kit-clang-options-doc
        kit-flang-options-doc
        copy-html-inputs)
      add_dependencies(kitsune-doc docs-kitsune-html)

      # docs-kitsune-html is generated by add_sphinx_target. Create an alias
      # for consistency with other documentation-related targets specific to
      # kitsune
      add_custom_target(kitsune-docs-html)
      add_dependencies(kitsune-docs-html docs-kitsune-html)

      if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
        # Install Kitsune's documentation to a directory that adheres to the GNU
        # coding standards. The documentation install directory for other LLVM
        # projects do not adhere to this standard. This change in Kitsune is
        # deliberate since we want both the doxygen-generated API documentation
        # and the HTML documentation to be installed to the same directory.
        #
        # ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/index.md is a placeholder file
        # that displays a message saying that doxygen-generated API
        # documentation is not available. If the doxygen-generated documentation
        # is not built, copy ${CMAKE_CURRENT_BINARY_DIR}/html/doxygen/
        # to doxygen/ in the destination. This way, we will have a valid link in
        # the table of contents since the index.md placeholder file will have
        # been used to generate doxygen/index.html.
        #
        # If doxygen-generated documentation will be built, do not copy
        # doxygen/index.html. In this case, the generated doxygen documentation
        # will be copied to doxygen/ in the destination directory. Doxygen
        # will have generated doxygen/index.html from the doxygen mainpage file.
        # This way, we will again have a link to valid page - but this time to
        # the landing page of the doxygen-generated documentation.
        #
        if (KITSUNE_ENABLE_DOXYGEN)
          install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/
            DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/kitsune/www/
            PATTERN doxygen/index.html EXCLUDE)
        else ()
          install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/
            DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/kitsune/www/)
        endif()
      endif()
    endif()

    if (SPHINX_OUTPUT_MAN)
      add_sphinx_target(man kitsune SOURCE_DIR "${MAN_DIR}"
        ENV_VARS "${ENV_VARS}")

      gen_options_rst(${MAN_DIR}/KitClangOptionsMan.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitClangOptionsMan.td
        kit-clang-options-man)
      add_dependencies(kit-clang-options-man kitsune-man-dir)

      gen_options_rst(${MAN_DIR}/KitFlangOptionsMan.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitFlangOptionsMan.td
        kit-flang-options-man)
      add_dependencies(kit-flang-options-man kitsune-man-dir)

      add_custom_target(copy-man-inputs
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/conf.py ${MAN_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/man/*.rst ${MAN_DIR}
        DEPENDS kitsune-man-dir)

      add_dependencies(docs-kitsune-man
        kit-clang-options-man
        kit-flang-options-man
        copy-man-inputs)
      add_dependencies(kitsune-doc docs-kitsune-man)

      # docs-kitsune-man is generated by add_sphinx_target. Create an alias
      # for consistency with other documentation-related targets specific to
      # kitsune
      add_custom_target(kitsune-docs-man)
      add_dependencies(kitsune-docs-man docs-kitsune-man)

      # If all of LLVM's documentation is not being built, install Kitsune's
      # documentation to a different location. The destination directory in
      # this case adheres to the GNU coding standards.
      if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/man/
          DESTINATION ${CMAKE_INSTALL_MANDIR}/man1)
      endif()
    endif()
  endif()
endif()
