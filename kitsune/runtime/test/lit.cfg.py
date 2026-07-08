import os
import platform
import re
import subprocess
import tempfile

import lit.formats

from lit.llvm import llvm_config
from lit.llvm.subst import FindTool
from lit.llvm.subst import ToolSubst

# name: The name of this test suite.
config.name = "Kitrt"

# The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# A list of file extensions to treat as test files.
config.suffixes = [".c", ".cpp", ".test"]

# Exclude some files and directories. The top-level test/ directory has a
# CMakeLists.txt file which is not needed. It looks like the README.md files are
# automatically ignored.
config.excludes = [
    "CMakeLists.txt",
    "input"
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.join(config.kitrt_source_dir, "test")

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.kitrt_binary_dir, "test")

llvm_config.use_default_substitutions()

# The %exe substitution computes the name of the executable from the name of the
# source file it is contained in. The executable is always found in
# `${KITSUNE_BINARY_DIR}/test/<subdir>/<filename>.test`. For instance, for the
# file ${KITSUNE_SOURCE_DIR}/runtime/test/openmp/ompNumThreadsTest.cpp, %exe
# must be ${KITSUNE_BINARY_DIR}/runtime/test/openmp/ompNumThreadsTest.cpp.test`.
# The commands `basename` and `dirname` will only be available on POSIX-ish
# systems, but those are the only systems that we currently support.
config.substitutions.append(
    ("%exe",
     f'{config.test_exec_root}/$(basename $(dirname %s))/$(basename %s).test'))

# For each occurrence of an LLVM tool name, replace it with the full path to
# the build directory holding that tool. We explicitly specify the directories
# to search to ensure that we get the tools just built and not some random
# tools that might happen to be in the user's PATH.
tool_dirs = [config.llvm_tools_dir]
tools = [ "not" ]
llvm_config.add_tool_substitutions(tools, tool_dirs)

# Features. We need the registered target features because the underlying
# runtimes used by certain tapir targets are only available on some platforms.
def calculate_arch_features(arch_string):
    features = []
    for arch in arch_string.split():
        features.append(arch.lower() + "-registered-target")
    return features

if config.kitsune_papi_enabled:
    config.available_features.add("kitsune-papi")

if config.kitsune_cuda_enabled:
    config.available_features.add("kitsune-cuda")

if config.kitsune_hip_enabled:
    config.available_features.add("kitsune-hip")

if config.kitsune_lambda_enabled:
    config.available_features.add("kitsune-lambda")

if config.kitsune_omptask_enabled:
    config.available_features.add("kitsune-omptask")

if config.kitsune_opencilk_enabled:
    config.available_features.add("kitsune-opencilk")

if config.kitsune_qthreads_enabled:
    config.available_features.add("kitsune-qthreads")

if config.kitsune_realm_enabled:
    config.available_features.add("kitsune-realm")
