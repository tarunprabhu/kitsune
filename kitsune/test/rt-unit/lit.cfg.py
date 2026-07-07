import os
import subprocess

import lit.formats

# The name of this test suite.
config.name = "kitrt-Unit"

# A list of file extensions to treat as test files.
config.suffixes = []

# The root path where tests are located.
config.test_source_root = os.path.join(config.kitrt_obj_root, "unittests")

# The root path where tests should be run.
config.test_exec_root = os.path.join(config.kitrt_obj_root, "unittests")

# The test format to use to interpret tests.
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, "Tests")

# Propagate the temp directory. Windows requires this because it uses \Windows\
# if none of these are present.
if "TMP" in os.environ:
    config.environment["TMP"] = os.environ["TMP"]
if "TEMP" in os.environ:
    config.environment["TEMP"] = os.environ["TEMP"]

# Propagate HOME because this may be used in places that are not at all obvious.
if "HOME" in os.environ:
    config.environment["HOME"] = os.environ["HOME"]
