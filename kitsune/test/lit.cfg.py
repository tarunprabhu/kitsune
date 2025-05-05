import os
import platform
import re
import subprocess
import tempfile

import lit.formats

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# name: The name of this test suite.
config.name = "Kitsune"

# The test format to use to interpret tests. The comment below was copied
# verbatim from clang's configuration. I have no idea what it means.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# A list of file extensions to treat as test files. We could probably reduce
# this list somewhat and spread it out among the lit.local files since some of
# these are only used in certain subdirectories.
config.suffixes = [
    ".c",
    ".cpp",
    ".cl",
    ".cu",
    ".hip",
    ".ll",
    ".f90",
    ".f",
    ".m",
    ".objc",
    ".s",
    ".S",
    ".test",
    ".rs",
    ".ifs",
    ".rc",
]

# Exclude some files and directories. The top-level test/ directory has a
# CMakeLists.txt file which is needed. It looks like the README.md files are
# automatically ignored.
config.excludes = [
    "CMakeLists.txt",
    "input"
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.kitsune_obj_root, "test")

llvm_config.use_default_substitutions()

llvm_config.use_clang()
llvm_config.use_lld()

config.substitutions.append(
    ("%src_include_dir", config.kitsune_src_dir + "/include")
)

config.substitutions.append(("%target_triple", config.target_triple))

config.substitutions.append(("%PATH%", config.environment["PATH"]))

# For each occurrence of a clang tool name, replace it with the full path to
# the build directory holding that tool.  We explicitly specify the directories
# to search to ensure that we get the tools just built and not some random
# tools that might happen to be in the user's PATH.
tool_dirs = [config.llvm_tools_dir]

tools = [
    "clang-linker-wrapper",
    "not",
    "opt",
    "llvm-dis",
    "llvm-lto",
    "llvm-lto2",
]
if config.kitsune_fortran_enabled:
    t = ToolSubst("%flang", command=FindTool("flang"), unresolved="fatal")
    tools.append(t)

llvm_config.add_tool_substitutions(tools, tool_dirs)

config.substitutions.append(("%host_cc", config.host_cc))
config.substitutions.append(("%host_cxx", config.host_cxx))
config.substitutions.append(("%kitcc", config.kitcc))
config.substitutions.append(("%kitxx", config.kitxx))
config.substitutions.append(("%kitfc", config.kitfc))
config.substitutions.append(("%kitconf", config.kitconf))
config.substitutions.append(("%kitsune_gcc_install_dir",
                             config.kitsune_gcc_install_dir))
if config.kitsune_sysroot:
    config.substitutions.append(("%sysroot",
                                 "--sysroot=" + config.kitsune_sysroot))
else:
    config.substitutions.append(("%sysroot", ""))

# Features. We need the registered target features because some runtimes used by
# the tapir targets will only run on certain architectures, so we need to
# conditionally enable those tests.
def calculate_arch_features(arch_string):
    features = []
    for arch in arch_string.split():
        features.append(arch.lower() + "-registered-target")
    return features

# This was copied from clang's lit.cfg.py, I think. I have no idea what this is
# doing.
llvm_config.feature_config([
    ("--assertion-mode", {
        "ON": "asserts"
    }), ("--cxxflags", {
        r"-D_GLIBCXX_DEBUG\b": "libstdcxx-safe-mode"
    }), ("--targets-built", calculate_arch_features),
])

if config.kitsune_gcc_install_dir:
    config.available_features.add("kitsune-gcc-install-dir")

if config.kitsune_c_enabled:
    config.available_features.add("kitcc")

if config.kitsune_cxx_enabled:
    config.available_features.add("kitxx")

if config.kitsune_fortran_enabled:
    config.available_features.add("kitfc")

# If these features are not enabled, create a corresponding no-<FEATURE>. This
# is needed to run tests that check the frontends handle the case where
# --tapir=<TARGET> is given but TARGET has not been enabled in the build.
if config.kitsune_kokkos_enabled:
    config.available_features.add("kitsune-kokkos")
else:
    config.available_features.add("kitsune-no-kokkos")

if config.kitsune_cuda_enabled:
    config.available_features.add("kitsune-cuda")
else:
    config.available_features.add("kitsune-no-cuda")

if config.kitsune_hip_enabled:
    config.available_features.add("kitsune-hip")
else:
    config.available_features.add("kitsune-no-hip")

if config.kitsune_lambda_enabled:
    config.available_features.add("kitsune-lambda")
else:
    config.available_features.add("kitsune-no-lambda")

if config.kitsune_omptask_enabled:
    config.available_features.add("kitsune-omptask")
else:
    config.available_features.add("kitsune-no-omptask")

if config.kitsune_opencilk_enabled:
    config.available_features.add("kitsune-opencilk")
else:
    config.available_features.add("kitsune-no-opencilk")

if config.kitsune_openmp_enabled:
    config.available_features.add("kitsune-openmp")
else:
    config.available_features.add("kitsune-no-openmp")

if config.kitsune_qthreads_enabled:
    config.available_features.add("kitsune-qthreads")
else:
    config.available_features.add("kitsune-no-qthreads")

if config.kitsune_realm_enabled:
    config.available_features.add("kitsune-realm")
else:
    config.available_features.add("kitsune-no-realm")

# It is not realistically possible to account for all options that could
# possibly be present in system and user configuration files, so disable
# default configs for the test runs.
config.environment["CLANG_NO_DEFAULT_CONFIG"] = "1"
