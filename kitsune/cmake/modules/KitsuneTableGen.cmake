include(TableGen)

# Run llvm-tblgen on a .td file SOURCE and generate the file DEST. The function
# takes an optional set of named arguments:
#
#   TARGET      A single-valued argument that is the name of a cmake target to
#               create for the .inc file created by this invocation of
#               llvm-tblgen. intrinsics_gen will be made to depend on this
#               target. This is solely to ensure that Kitsune-generated .inc
#               files are created before any files in llvm/lib are compiled.
#
#   DEPENDS     A list of cmake targets that this invocation depends on.
#
# Any addition arguments that are not associated with any of these named
# arguments will be passed as-is to llvm-tblgen
#
# EXAMPLE
#
#     kitsune_tablegen(Source.td Dest.inc
#                      -gen-kitsune-loop-attrs -I/path/to/extra/include
#                      TARGET kitsune_loop_attrs_gen)
#
# In this invocation, -gen-kitsune-loop-attrs -I/path/to/extra/include will be
# passed to llvm-tblgen.
#
function(kitsune_tablegen source dest)
  cmake_parse_arguments(ARGS "" "TARGET" "DEPENDS" ${ARGN})

  # The tablegen function below is what is used by LLVM and requires
  # LLVM_TARGET_DEFINITIONS to be set to the source tablegen file.
  set(LLVM_TARGET_DEFINITIONS ${source})

  # The prefix LLVM here is used to construct the cmake variable
  # LLVM_TABLEGEN_EXE, which contains the path to the llvm-tblgen executable. At
  # some point, we should probably set a corresponding KITSUNE_TABLEGEN_EXE
  # variable and use that instead.
  tablegen(LLVM ${dest} ${ARGS_UNPARSED_ARGUMENTS})

  if(ARGS_TARGET)
    add_public_tablegen_target(${ARGS_TARGET})
  endif()
endfunction(kitsune_tablegen)
