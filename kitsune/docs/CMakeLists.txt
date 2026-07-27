include(KitsuneTableGen)

add_custom_target(kitsune-doc)

# This is a convenience synonym for kitsune-doc. The latter is present for
# consistency with other LLVM subprojects.
add_custom_target(kitsune-docs)
add_dependencies(kitsune-docs kitsune-doc)

# Generate a .rst file from a .td file from a Kitsune-specific tablegen file.
#
# RST         The output .rst file
# TD          The input .td file from which the .rst file is generated
# COMMAND     The command to pass to llvm-tblgen
# TARGET_RST  The custom target that builds the .rst file. The target itself
#             will be created in this function
# TARGET_DIR  The name of the custom target that makes the directory into which
#             the .rst file will be written. This target should have been
#             created before this function is called
function(gen_kit_rst rst td command target_rst target_dir)
  kitsune_tablegen(${td} ${rst} ${command}
    TARGET ${target_rst})

  add_dependencies(${target_rst} ${target_dir})
endfunction ()

# Generate a .rst file from a .td file that serves as documentation for
# Kitsune's C/C++ attributes.
#
# RST         The output .rst file
# TD          The input .td file from which the .rst file is generated
# TARGET_RST  The custom target that builds the .rst file. The target itself
#             will be created in this function
# TARGET_DIR  The name of the custom target that makes the directory into which
#             the .rst file will be written. This target should have been
#             created before this function is called.
function (gen_attrs_rst rst td target_rst target_dir)
  get_filename_component(TABLEGEN_INCLUDE_DIR ${td} DIRECTORY)
  list(APPEND LLVM_TABLEGEN_FLAGS
    "-I${TABLEGEN_INCLUDE_DIR}"
    "-I${LLVM_MONOREPO_SOURCE_DIR}/clang/include"
    "-I${LLVM_MONOREPO_SOURCE_DIR}/kitsune/include")
  clang_tablegen(${rst} -gen-kitsune-attr-docs
    SOURCE ${td}
    TARGET ${target_rst})
  add_dependencies(${target_rst} ${target_dir})
endfunction()

# Generate a .rst file from a .td file that serves as documentation for
# command-line options supported by Kitsune's drivers.
#
# RST         The output .rst file
# TD          The input .td file from which the .rst file is generated
# TARGET_RST  The custom target that builds the .rst file. The target itself
#             will be created in this function
# TARGET_DIR  The name of the custom target that makes the directory into which
#             the .rst file will be written. This target should have been
#             created before this function is called.
function (gen_options_rst rst td target_rst target_dir)
  get_filename_component(TABLEGEN_INCLUDE_DIR ${td} DIRECTORY)
  list(APPEND LLVM_TABLEGEN_FLAGS "-I${TABLEGEN_INCLUDE_DIR}")
  list(APPEND LLVM_TABLEGEN_FLAGS
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../clang/include/clang/Driver/")
  clang_tablegen(${rst} -gen-kitsune-opt-docs
    SOURCE ${td}
    TARGET ${target_rst})
  add_dependencies(${target_rst} ${target_dir})
endfunction()

# Shorter variables for cmake -E commands.
set(MAKE_DIRECTORY ${CMAKE_COMMAND} -E make_directory)
set(COPY_IF_DIFFERENT ${CMAKE_COMMAND} -E copy_if_different)
set(COPY_DIR_IF_DIFFERENT ${CMAKE_COMMAND} -E copy_directory_if_different)

if (KITSUNE_ENABLE_DOXYGEN OR KITSUNE_ENABLE_SPHINX)
  # The directory where *all* HTML documentation - including the Doxygen
  # documentation, if enabled - will be collected. The contents of this
  # directory are suitable to be served by an HTTP server.
  set(HTML_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/html)
  add_custom_target(kitsune-html-output-dir
    COMMAND ${MAKE_DIRECTORY} ${HTML_OUTPUT_DIR}
    USES_TERMINAL VERBATIM)
endif ()

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
    COMMENT "Generating kitsune doxygen documentation"
    DEPENDS kitsune-html-output-dir
    USES_TERMINAL VERBATIM)
  set_target_properties(doxygen-kitsune PROPERTIES FOLDER "Kitsune/Docs")

  # Create an alias for doxygen-kitsune to be consistent with the other
  # documentation-related targets specific to Kitsune
  add_custom_target(kitsune-docs-doxygen)
  add_dependencies(kitsune-docs-doxygen doxygen-kitsune)

  add_dependencies(kitsune-docs kitsune-docs-doxygen)
  if (LLVM_BUILD_DOCS)
   add_dependencies(doxygen kisune-docs-doxygen)
  endif()
endif ()

if (KITSUNE_ENABLE_SPHINX)
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
    if (KITSUNE_ENABLE_DOXYGEN)
      set(KITSUNE_API_REFERENCE_LINK "doxygen/html/index.html")
    else ()
      set(KITSUNE_API_REFERENCE_LINK "doxygen/APIReferenceUnavailable.html")
    endif ()

    set(ENV_VARS
      CMAKE_SOURCE_DIR="${CMAKE_SOURCE_ROOT_DIR}"
      KITSUNE_API_REFERENCE_LINK="${KITSUNE_API_REFERENCE_LINK}"
      KITSUNE_REPO_HOST="${KITSUNE_REPO_HOST}"
      KITSUNE_REPO_OWNER="${KITSUNE_REPO_OWNER}"
      KITSUNE_REPO_BRANCH="${KITSUNE_REPO_BRANCH}"
      KITSUNE_DEFAULT_LANGS="${KITSUNE_DEFAULT_LANGS}"
      KITSUNE_UNIVERSAL_TAPIR_TARGETS="${KITSUNE_UNIVERSAL_TAPIR_TARGETS}"
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

    if (SPHINX_OUTPUT_HTML)
      # The directory to which the .md files from
      # ${CMAKE_SOURCE_DIR}/kitsune/docs are copied. This will also contain the
      # .rst files generated from .td files. The is the input to sphinx - all
      # files in the directory will be converted to HTML.
      set(HTML_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/SourceHtml)
      add_custom_target(kitsune-html-source-dir
        COMMAND ${MAKE_DIRECTORY} ${HTML_SOURCE_DIR}
        USES_TERMINAL VERBATIM)

      # The environment variables in ${ENV_VARS} are read directly in conf.py.
      # sphinx-build accepts a SPHINXOPT environment variable, but it seems as
      # if any -D options in that variable are not recognized.
      add_sphinx_target(html kitsune SOURCE_DIR "${HTML_SOURCE_DIR}"
        ENV_VARS "${ENV_VARS}")

      gen_attrs_rst(${HTML_SOURCE_DIR}/KitAttrsDoc.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitAttrsDoc.td
        kitsune-attrs-doc
        kitsune-html-source-dir)

      gen_kit_rst(${HTML_SOURCE_DIR}/KitCBuiltinsDoc.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitCBuiltinsDoc.td
        -gen-kit-cbuiltins-doc
        kitsune-cbuiltins-doc
        kitsune-html-source-dir)

      gen_kit_rst(${HTML_SOURCE_DIR}/KitIntrinsicsDoc.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitIntrinsicsDoc.td
        -gen-kit-intrinsics-doc
        kitsune-intrinsics-doc
        kitsune-html-source-dir)

      gen_kit_rst(${HTML_SOURCE_DIR}/KitArgAttrsDoc.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitArgAttrsDoc.td
        -gen-kit-arg-attr-doc
        kitsune-arg-attrs-doc
        kitsune-html-source-dir)

      gen_kit_rst(${HTML_SOURCE_DIR}/KitFuncAttrsDoc.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitFuncAttrsDoc.td
        -gen-kit-func-attr-doc
        kitsune-func-attrs-doc
        kitsune-html-source-dir)

      gen_kit_rst(${HTML_SOURCE_DIR}/KitGVAttrsDoc.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitGVAttrsDoc.td
        -gen-kit-gv-attr-doc
        kitsune-gv-attrs-doc
        kitsune-html-source-dir)

      gen_kit_rst(${HTML_SOURCE_DIR}/KitInstAttrsDoc.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitInstAttrsDoc.td
        -gen-kit-inst-attr-doc
        kitsune-inst-attrs-doc
        kitsune-html-source-dir)

      gen_kit_rst(${HTML_SOURCE_DIR}/KitLoopAttrsDoc.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitLoopAttrsDoc.td
        -gen-kit-loop-attr-doc
        kitsune-loop-attrs-doc
        kitsune-html-source-dir)

      gen_kit_rst(${HTML_SOURCE_DIR}/KitModuleAttrsDoc.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitModuleAttrsDoc.td
        -gen-kit-module-attr-doc
        kitsune-module-attrs-doc
        kitsune-html-source-dir)

      gen_options_rst(${HTML_SOURCE_DIR}/KitClangOptionsDoc.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitClangOptionsDoc.td
        kitsune-clang-options-doc
        kitsune-html-source-dir)

      gen_options_rst(${HTML_SOURCE_DIR}/KitCC1OptionsDoc.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitCC1OptionsDoc.td
        kitsune-cc1-options-doc
        kitsune-html-source-dir)

      gen_options_rst(${HTML_SOURCE_DIR}/KitFlangOptionsDoc.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitFlangOptionsDoc.td
        kitsune-flang-options-doc
        kitsune-html-source-dir)

      gen_options_rst(${HTML_SOURCE_DIR}/KitFC1OptionsDoc.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitFC1OptionsDoc.td
        kitsune-fc1-options-doc
        kitsune-html-source-dir)

      set(MDS
        AddingCommandLineOption.md
        AddingKitsuneIntrinsic.md
        AddingTapirTarget.md
        BasicUsage.md
        BuildingDocumentation.md
        BuildSystem.md
        CodeOrganization.md
        CommandLineOptions.md
        ConfigurationFiles.md
        DriverDesign.md
        EmbeddedBitcode.md
        FAQ.md
        FortranSupport.md
        GettingStarted.md
        Glossary.md
        index.md
        KitInstructionsDoc.md
        KitPassesDoc.md
        KitsuneTestSuite.md
        KokkosSupport.md
        LanguageExtensions.md
        LLVMIRAttributes.md
        LLVMTools.md
        LTO.md
        MemoryManagement.md
        OldReleaseNotes.md
        Overview.md
        PassPipeline.md
        RegisteringLibraryFunction.md
        ReleaseNotes.md
        StaticLinking.md
        TapirPassesDoc.md
        TapirTargets.md
        Testing.md
        WritingEmbeddedBitcodePass.md
        WritingPassPlugin.md
        WritingTapirTargetPlugin.md)

      add_custom_target(copy-html-inputs
        COMMAND ${COPY_IF_DIFFERENT} conf.py ${HTML_SOURCE_DIR}
        COMMAND ${COPY_IF_DIFFERENT} ${MDS} ${HTML_SOURCE_DIR}
        COMMAND ${COPY_DIR_IF_DIFFERENT} CommandGuide ${HTML_SOURCE_DIR}/CommandGuide
        COMMAND ${COPY_DIR_IF_DIFFERENT} _themes ${HTML_SOURCE_DIR}/_themes
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS kitsune-html-source-dir
        USES_TERMINAL VERBATIM)

      add_dependencies(docs-kitsune-html
        kitsune-attrs-doc
        kitsune-cbuiltins-doc
        kitsune-clang-options-doc
        kitsune-cc1-options-doc
        kitsune-flang-options-doc
        kitsune-fc1-options-doc
        kitsune-arg-attrs-doc
        kitsune-func-attrs-doc
        kitsune-gv-attrs-doc
        kitsune-inst-attrs-doc
        kitsune-intrinsics-doc
        kitsune-loop-attrs-doc
        kitsune-module-attrs-doc
        copy-html-inputs)

      # If Kitsune's API reference is to be built, the doxygen/ directory will
      # be contain the generated documents. Otherwise, copy the contents of
      # ${CMAKE_CURRENT_SOURCE_DIR}/doxygen. This will contain a single file
      # that displays an error message saying that Kitsune's API reference is
      # unavailable.
      if (KITSUNE_ENABLE_DOXYGEN)
        add_dependencies(docs-kitsune-html
          kitsune-docs-doxygen)
      else ()
        add_custom_target(copy-api-reference-unavailable
          COMMAND ${COPY_DIR_IF_DIFFERENT} doxygen ${HTML_SOURCE_DIR}/doxygen
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          DEPENDS kitsune-html-source-dir
          USES_TERMINAL VERBATIM)
        add_dependencies(docs-kitsune-html
          copy-api-reference-unavailable)
      endif ()

      add_dependencies(kitsune-doc docs-kitsune-html)

      # docs-kitsune-html is generated by add_sphinx_target. Create an alias
      # for consistency with other documentation-related targets specific to
      # kitsune.
      add_custom_target(kitsune-docs-html)
      add_dependencies(kitsune-docs-html docs-kitsune-html)

      if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
        # Install Kitsune's documentation to a directory that adheres to the GNU
        # coding standards. The documentation install directory for other LLVM
        # projects do not adhere to this standard. This change in Kitsune is
        # deliberate since we want both the doxygen-generated API documentation
        # and the HTML documentation to be installed to the same directory.
        #
        # NOTE: None of the documentation generation targets are added to ALL,
        # so they must be run manually before running install. Issue a warning
        # if the directory does not exist because the install(DIRECTORY) command
        # will silently skip the directory if it has not been generated.
        get_filename_component(MAKE "${CMAKE_MAKE_PROGRAM}" NAME)
        install(CODE "
            if (NOT EXISTS \"${CMAKE_CURRENT_BINARY_DIR}/html\")
                message(WARNING
\"Kitsune's HTML documentation has not been generated.
Run `${MAKE} kitsune-docs-html` (or `${MAKE} kitsune-docs`) before installing\")
            endif ()")
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/
          DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/kitsune/www/
          OPTIONAL)
      endif()
    endif()

    if (SPHINX_OUTPUT_MAN)
      # The directory containing the .rst files for man pages generated from
      # .td files. The actual man pages will be generated from these .rst files.
      # The directory is the input to sphinx - all files in it will be converted
      # to HTML.
      set(MAN_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/SourceMan)
      add_custom_target(kitsune-man-source-dir
        COMMAND ${MAKE_DIRECTORY} ${MAN_SOURCE_DIR}
        USES_TERMINAL VERBATIM)

      # The environment variables in ${ENV_VARS} are read directly in conf.py.
      # sphinx-build accepts a SPHINXOPT environment variable, but it seems as
      # if any -D options in that variable are not recognized.
      add_sphinx_target(man kitsune SOURCE_DIR "${MAN_SOURCE_DIR}"
        ENV_VARS "${ENV_VARS}")

      gen_options_rst(${MAN_SOURCE_DIR}/KitClangOptionsMan.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitClangOptionsMan.td
        kitsune-clang-options-man
        kitsune-man-source-dir)

      gen_options_rst(${MAN_SOURCE_DIR}/KitFlangOptionsMan.rst
        ${CMAKE_CURRENT_SOURCE_DIR}/KitFlangOptionsMan.td
        kitsune-flang-options-man
        kitsune-man-source-dir)

      set(RSTS
        man/index.rst
        man/kitcc.rst
        man/kit++.rst
        man/kitfc.rst)

      add_custom_target(copy-man-inputs
        COMMAND ${COPY_IF_DIFFERENT} conf.py ${MAN_SOURCE_DIR}
        COMMAND ${COPY_IF_DIFFERENT} ${RSTS} ${MAN_SOURCE_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS kitsune-man-source-dir
        USES_TERMINAL VERBATIM)

      add_dependencies(docs-kitsune-man
        kitsune-clang-options-man
        kitsune-flang-options-man
        copy-man-inputs)
      add_dependencies(kitsune-doc docs-kitsune-man)

      # docs-kitsune-man is generated by add_sphinx_target. Create an alias for
      # consistency with other documentation-related targets specific to
      # kitsune.
      add_custom_target(kitsune-docs-man)
      add_dependencies(kitsune-docs-man docs-kitsune-man)

      # If all of LLVM's documentation is not being built, install Kitsune's
      # documentation to a different location. The destination directory in
      # this case adheres to the GNU coding standards.
      if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
        # NOTE: None of the documentation generation targets are added to ALL,
        # so they must be run manually before running install. Issue a warning
        # if the directory does not exist because the install(DIRECTORY) command
        # will silently skip the directory if the documentation has not been
        # generated.
        get_filename_component(MAKE "${CMAKE_MAKE_PROGRAM}" NAME)
        install(CODE "
            if (NOT EXISTS \"${CMAKE_CURRENT_BINARY_DIR}/man\")
                message(WARNING
\"Kitsune's man pages have not been generated.
Run `${MAKE} kitsune-docs-man` (or `${MAKE} kitsune-docs`) before installing\")
            endif ()")
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/man/
          DESTINATION ${CMAKE_INSTALL_MANDIR}/man1
          OPTIONAL)
      endif()
    endif()
  endif()
endif()
