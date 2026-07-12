import os

import lit.formats

from lit.llvm import llvm_config
from lit.llvm.subst import FindTool
from lit.llvm.subst import ToolSubst

# The name of this test suite.
config.name = "kitsune"

# The test format to use to interpret tests.
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
# CMakeLists.txt file which is not needed. It looks like the README.md files are
# automatically ignored.
config.excludes = [
    "CMakeLists.txt",
    "input"
]

# The root path where tests are located.
config.test_source_root = os.path.join(config.kitsune_source_dir, "test")

# The root path where tests should be run.
config.test_exec_root = os.path.join(config.kitsune_binary_dir, "test")

# It is not realistically possible to account for all options that could
# possibly be present in system and user configuration files, so disable
# default configuration files for the test runs.
config.environment["CLANG_NO_DEFAULT_CONFIG"] = "1"

llvm_config.use_default_substitutions()

config.substitutions.append(("%llvm_binary_dir", config.llvm_binary_dir))
config.substitutions.append(("%llvm_source_dir", config.llvm_source_dir))
config.substitutions.append(("%kitsune_gcc_install_dir",
                             config.kitsune_gcc_install_dir))
config.substitutions.append(("%kitsune_source_dir", config.kitsune_source_dir))
config.substitutions.append(("%kit-emb-pass-plugin-demo",
                             config.kitsune_emb_pass_plugin_demo))
config.substitutions.append(("%kit-pass-plugin-demo",
                             config.kitsune_pass_plugin_demo))
config.substitutions.append(("%kit-tt-plugin-demo",
                             config.kitsune_tt_plugin_demo))
config.substitutions.append(("%shlibext", config.shlibext))
if config.kitsune_sysroot:
    config.substitutions.append(("%sysroot",
                                 "--sysroot=" + config.kitsune_sysroot))
else:
    config.substitutions.append(("%sysroot", ""))

# For each occurrence of a clang tool name, replace it with the full path to
# the build directory holding that tool. We explicitly specify the directories
# to search to ensure that we get the tools just built and not some random
# tools that might happen to be in the user's PATH.
tool_dirs = [config.llvm_tools_dir]
tools = [
    r"ld.lld",
    r"ld64.lld",
    "llc",
    "llvm-as",
    "llvm-dis",
    "llvm-lto",
    "llvm-lto2",
    "not",
    "opt",
    "wasm-ld",
]
tools.extend([
    ToolSubst("%clang", FindTool("clang"), unresolved="fatal"),
    ToolSubst("%clangxx", FindTool("clang++"), unresolved="fatal"),
    ToolSubst("%kitcc", FindTool(config.kitcc), unresolved="fatal"),
    ToolSubst("%kitxx", FindTool(config.kitxx), unresolved="fatal"),
])
if config.kitsune_fortran_enabled:
    tools.extend([
        ToolSubst("%flang", FindTool("flang"), unresolved="fatal"),
        ToolSubst("%kitfc", FindTool(config.kitfc), unresolved="fatal"),
    ])
for tool in config.kitsune_tools.split(';'):
    tools.append(ToolSubst(f"%{tool}", FindTool(tool), unresolved="fatal"))

llvm_config.add_tool_substitutions(tools, tool_dirs)

if config.kitsune_gcc_install_dir:
    config.available_features.add("kitsune-gcc-install-dir")

if config.kitsune_build_examples:
    config.available_features.add("kitsune-examples")

if config.kitsune_c_enabled:
    config.available_features.add("kitcc")

if config.kitsune_cxx_enabled:
    config.available_features.add("kitxx")

if config.kitsune_fortran_enabled:
    config.available_features.add("kitfc")

# If these features are not enabled, create a corresponding no-<FEATURE>.
# The drivers should raise an error when --tapir=<TARGET> is provided on the
# command line and the tapir target <TARGET> has not been enabled in the build.
# In order to test this, we have to run some tests *only* when some tapir target
# <TARGET> has not been built. The "kitsune-no-*" features defined below are
# required for these tests.
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

if config.kitsune_qthreads_enabled:
    config.available_features.add("kitsune-qthreads")
else:
    config.available_features.add("kitsune-no-qthreads")

if config.kitsune_realm_enabled:
    config.available_features.add("kitsune-realm")
else:
    config.available_features.add("kitsune-no-realm")
