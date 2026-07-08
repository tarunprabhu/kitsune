import os
import subprocess

import lit.formats

# The name of this test suite.
config.name = "kitrt-unit"

# We don't need to specify any suffixes because this directory only contains
# GoogleTests and gtest will know how to discover them.
config.suffixes = []

# The root path where tests are located.
config.test_source_root = os.path.join(config.kitrt_binary_dir, "unittests")

# The root path where tests should be run.
config.test_exec_root = os.path.join(config.kitrt_binary_dir, "unittests")

# All GoogleTests are named to have 'Tests' as their suffix. The '.' option is
# a special value for GoogleTest indicating that it should look through the
# entire testsuite recursively for tests.
config.test_format = lit.formats.GoogleTest(".", "Tests")

# Propagate HOME because this may be used in places that are not at all obvious.
if "HOME" in os.environ:
    config.environment["HOME"] = os.environ["HOME"]
