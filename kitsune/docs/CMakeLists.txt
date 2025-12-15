add_custom_target(kitsune-doc)
set_target_properties(kitsune-doc PROPERTIES FOLDER "Kitsune/Docs")

# This is a convenience synonym for kitsune-doc. The latter is present for
# consistency with other LLVM subprojects.
add_custom_target(kitsune-docs)
add_dependencies(kitsune-docs kitsune-doc)

function (gen_rst_from_td rst td target)
  get_filename_component(TABLEGEN_INCLUDE_DIR ${td} DIRECTORY)
  list(APPEND LLVM_TABLEGEN_FLAGS "-I${TABLEGEN_INCLUDE_DIR}")
  list(APPEND LLVM_TABLEGEN_FLAGS
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../clang/include/clang/Driver/")
  clang_tablegen(${rst} -gen-kitsune-opt-docs
    SOURCE ${td}
    TARGET ${target})
endfunction()

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

  include(AddSphinxTarget)

  if (SPHINX_FOUND)
    # CLANG_TABLEGEN_EXE must be set for clang_tablegen to run without errors
    find_program(CLANG_TABLEGEN_EXE "clang-tblgen"
      PATHS ${LLVM_TOOLS_BINARY_DIR}
      NO_DEFAULT_PATH)

    # Create the output directories
    add_custom_target(kitsune-html-dir
      COMMAND ${CMAKE_COMMAND} -E make_directory ${HTML_DIR})
    add_custom_target(kitsune-man-dir
      COMMAND ${CMAKE_COMMAND} -E make_directory ${MAN_DIR})

    # Since the top-level CMakeLists.txt file is in the llvm/ directory of this
    # checkout, CMAKE_SOURCE_DIR will point to that directory. The true root is
    # one level up from there.
    get_filename_component(CMAKE_SOURCE_ROOT_DIR "${CMAKE_SOURCE_DIR}" DIRECTORY)

    if (SPHINX_OUTPUT_HTML)
      # The environment variables here are read directly in conf.py.
      # sphinx-build accepts a SPHINXOPT environment variable, but it seems as
      # if any -D options in that variable are not recognized.
      add_sphinx_target(html kitsune SOURCE_DIR "${HTML_DIR}"
        ENV_VARS
        SPHINXOPTS="-j 1"
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

      gen_rst_from_td(${HTML_DIR}/KitClangOptionsDoc.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitClangOptionsDoc.td
        kit-clang-options-doc)
      add_dependencies(kit-clang-options-doc kitsune-html-dir)

      gen_rst_from_td(${HTML_DIR}/KitFlangOptionsDoc.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitFlangOptionsDoc.td
        kit-flang-options-doc)
      add_dependencies(kit-flang-options-doc kitsune-html-dir)

      # Sphinx needs all the input files in one place for it to work. Since
      # some of the input files, such as the command-line option references,
      # are generated, they will be in the build directory. This means that the
      # static (non-generated) files also need to be copied to the build
      # directory before invoking sphinx-build.
      add_custom_target(copy-inputs
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/conf.py ${HTML_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/*.md ${HTML_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/CommandGuide ${HTML_DIR}/CommandGuide
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/static ${HTML_DIR}/static
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/_themes/kitsune-theme
        ${HTML_DIR}/_themes/kitsune-theme
        DEPENDS kitsune-html-dir)

      add_dependencies(docs-kitsune-html
        kit-clang-options-doc
        kit-flang-options-doc
        copy-inputs)
      add_dependencies(kitsune-doc docs-kitsune-html)
    endif()

    if (SPHINX_OUTPUT_MAN)
      add_sphinx_target(man kitsune SOURCE_DIR "${MAN_DIR}")

      gen_rst_from_td(${MAN_DIR}/KitClangOptionsMan.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitClangOptionsMan.td
        kit-clang-options-man)
      add_dependencies(kit-clang-options-man kitsune-man-dir)

      gen_rst_from_td(${MAN_DIR}/KitFlangOptionsMan.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitFlangOptionsMan.td
        kit-flang-options-man)
      add_dependencies(kit-flang-options-man kitsune-man-dir)

      add_custom_target(copy-conf-man
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/conf.py ${MAN_DIR}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/conf.py kitsune-man-dir)

      add_custom_target(copy-index-rst
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/man/index.rst ${MAN_DIR}
        DEPENDS man/index.rst kitsune-man-dir)

      add_custom_target(copy-kitcc-rst
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/man/kitcc.rst ${MAN_DIR}
        DEPENDS man/kitcc.rst kitsune-man-dir)

      add_custom_target(copy-kitxx-rst
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/man/kit++.rst ${MAN_DIR}
        DEPENDS man/kit++.rst kitsune-man-dir)

      add_custom_target(copy-kitfc-rst
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/man/kitfc.rst ${MAN_DIR}
        DEPENDS man/kitfc.rst kitsune-man-dir)

      add_dependencies(docs-kitsune-man
        kit-clang-options-man
        kit-flang-options-man
        copy-conf-man
        copy-index-rst
        copy-kitcc-rst
        copy-kitxx-rst
        copy-kitfc-rst)
      add_dependencies(kitsune-doc docs-kitsune-man)
    endif()
  endif()
endif()
