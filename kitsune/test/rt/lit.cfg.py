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

# The test format to use to interpret tests. The comment below was copied
# verbatim from clang's configuration. I have no idea what it means.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# A list of file extensions to treat as test files. We could probably reduce
# this list somewhat and spread it out among the lit.local files since some of
# these are only used in certain subdirectories.
config.suffixes = [".c", ".cpp", ".test"]

# Exclude some files and directories. The top-level test/ directory has a
# CMakeLists.txt file which is needed. It looks like the README.md files are
# automatically ignored.
config.excludes = [
    "CMakeLists.txt",
    "input"
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.join(config.kitrt_src_root, "test")

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.kitrt_obj_root, "test")

llvm_config.use_default_substitutions()

config.substitutions.append(("%b", os.path.join(config.kitrt_obj_root, "test")))
config.substitutions.append(
    ("%exe", f'{config.test_exec_root}/$(basename $(dirname %s))/$(basename %s).test'))
config.substitutions.append(("%PATH%", config.environment["PATH"]))

# For each occurrence of a clang tool name, replace it with the full path to
# the build directory holding that tool. We explicitly specify the directories
# to search to ensure that we get the tools just built and not some random
# tools that might happen to be in the user's PATH.
tool_dirs = [config.llvm_tools_dir]

tools = [ "not" ]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# Features. We need the registered target features because some runtimes used by
# the tapir targets will only run on certain architectures, so we need to
# conditionally enable those tests.
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
