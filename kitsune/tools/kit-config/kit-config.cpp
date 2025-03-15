//===-- kit-config.cpp - Kitsune configuration utility --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tool encapsulates information about Kitsune's configuration. This is
// an analog of the LLVM project's llvm-config utility, but the scope is very
// narrow. For most things, llvm-config should be used.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Config/config.h"
#include "llvm/Config/config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static void usage(bool exitWithFailure = true) {
  errs() << "\
usage: kit-config <OPTION>... \n\
\n\
Get configuration information about Kitsune\n\
\n\
Options:\n\
  --all-langs          All languages that have a frontend\n\
  --all-tapir-targets  The names of all known tapir targets\n\
  --c                  Has the C frontend been built (ON or OFF)\n\
  --c-frontend         Path to Kitsune's C frontend\n\
  --cxx                Has the C++ frontend been built (ON or OFF)\n\
  --cxx-frontend       Path to Kitsune's C++ frontend\n\
  --cuda-prefix        The cuda install prefix used by the Cuda Tapir target\n\
  --cuda-target        Has the Cuda Tapir target been built (ON or OFF)\n\
  --fortran            Has the Fortran frontend been built (ON or OFF)\n\
  --fortran-frontend   Path to Kitsune's Fortran frontend\n\
  --help               Print a summary of kit-config arguments\n\
  --hip-prefix         The rocm install prefix used by the Hip Tapir target\n\
  --hip-target         Has the Hip Tapir target been built (ON or OFF)\n\
  --kitsune-version    Print Kitsune version\n\
  --kokkos-mode        Is Kokkos mode enabled (ON or OFF)\n\
  --lambda-target      Has the Lambda tapir target been built (ON or OFF)\n\
  --langs              List all languages for which a frontend has been built\n\
  --llvm-version       Print LLVM version on which this is based\n\
  --omptask-target     Has the OMPTask tapir target been built (ON or OFF)\n\
  --opencilk-target    Has the OpenCilk tapir target been built (ON or OFF)\n\
  --openmp-target      Has the OpenMP tapir target been built (ON or OFF)\n\
  --qthreads-target    Has the Qthreads tapir target been built (ON or OFF)\n\
  --realm-target       Has the Realm tapir target been built (ON or OFF)\n\
  --tapir-targets      List all tapir targets that have been built\n\
  --version            Prints both LLVM and Kitsune versions\n\
\n\
When querying paths to frontends and prefixes, no output will be printed if\n\
if the corresponding frontend or related tapir target has not been built\n";
  if (exitWithFailure)
    exit(1);
}

// Get the path to the given frontend. @ref kitConfig is the full path to the
// this kitConfig executable. The frontend is assumed to be in the same
// directory as kitConfig.
static std::string frontendPath(StringRef kitConfig, StringRef frontend) {
  SmallString<256> path(sys::path::parent_path(kitConfig));
  sys::path::append(path, frontend);

  return std::string(path);
}

static void render(bool b) { outs() << (b ? "ON" : "OFF") << "\n"; }

static void render(const char *s) { outs() << s << "\n"; }

static void renderIf(bool cond, StringRef s) {
  if (cond)
    outs() << s << "\n";
}

static void renderVersions() {
  outs() << "LLVM version: " << PACKAGE_VERSION << "\n";
  outs() << "Kitsune version: " << KITSUNE_PACKAGE_VERSION << "\n";
}

static void renderPathIf(bool cond, const char *argv0, StringRef base) {
  if (cond) {
    std::string exe = sys::fs::getMainExecutable(argv0, (void *)usage);
    outs() << frontendPath(exe, base) << "\n";
  }
}

int main(int argc, char **argv) {
  if (argc == 1)
    usage();

  for (int i = 1; i < argc; ++i) {
    StringRef arg = argv[i];
    if (arg == "--help")
      usage(false);
    else if (arg == "--all-langs")
      render(KITSUNE_ALL_LANGS);
    else if (arg == "--all-tapir-targets")
      render(KITSUNE_ALL_TAPIR_TARGETS);
    else if (arg == "--c")
      render((bool)KITSUNE_C_ENABLED);
    else if (arg == "--c-frontend")
      renderPathIf(KITSUNE_C_ENABLED, argv[0], KITSUNE_C_FRONTEND);
    else if (arg == "--cxx")
      render((bool)KITSUNE_CXX_ENABLED);
    else if (arg == "--cxx-frontend")
      renderPathIf(KITSUNE_CXX_ENABLED, argv[0], KITSUNE_CXX_FRONTEND);
    else if (arg == "--cuda-prefix")
      renderIf(KITSUNE_CUDA_ENABLED, KITSUNE_CUDA_PREFIX);
    else if (arg == "--cuda-target")
      render((bool)KITSUNE_CUDA_ENABLED);
    else if (arg == "--fortran")
      render((bool)KITSUNE_Fortran_ENABLED);
    else if (arg == "--fortran-frontend")
      renderPathIf(KITSUNE_Fortran_ENABLED, argv[0], KITSUNE_Fortran_FRONTEND);
    else if (arg == "--hip-prefix")
      renderIf(KITSUNE_HIP_ENABLED, KITSUNE_HIP_PREFIX);
    else if (arg == "--hip-target")
      render((bool)KITSUNE_HIP_ENABLED);
    else if (arg == "--kitsune-version")
      render(KITSUNE_PACKAGE_VERSION);
    else if (arg == "--kokkos")
      render((bool)KITSUNE_KOKKOS_ENABLED);
    else if (arg == "--lambda-target")
      render((bool)KITSUNE_LAMBDA_ENABLED);
    else if (arg == "--langs")
      render(KITSUNE_ENABLED_LANGS);
    else if (arg == "--llvm-version")
      render(PACKAGE_VERSION);
    else if (arg == "--omptask-target")
      render((bool)KITSUNE_OMPTASK_ENABLED);
    else if (arg == "--opencilk-target")
      render((bool)KITSUNE_OPENCILK_ENABLED);
    else if (arg == "--openmp-target")
      render((bool)KITSUNE_OPENMP_ENABLED);
    else if (arg == "--qthreads-target")
      render((bool)KITSUNE_REALM_ENABLED);
    else if (arg == "--realm-target")
      render((bool)KITSUNE_REALM_ENABLED);
    else if (arg == "--tapir-targets")
      render(KITSUNE_ENABLED_TAPIR_TARGETS);
    else if (arg == "--version")
      renderVersions();
    else
      usage();
  }
  return 0;
}
