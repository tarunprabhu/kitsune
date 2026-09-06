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

#include "kitsune/Config/Config.h"
#include "kitsune/Core/TTUtils.h"
#include "llvm/ADT/StringExtras.h"
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
  --c                   Has the C frontend been built (ON or OFF)\n\
  --c-frontend          Path to the C frontend\n\
  --cuda-prefix         The cuda installation used by the cuda tapir target\n\
  --cuda-target         Has the cuda tapir target been built (ON or OFF)\n\
  --custom-target       Has the custom tapir target been built (ON or OFF)\n\
  --cxx                 Has the C++ frontend been built (ON or OFF)\n\
  --cxx-frontend        Path to the C++ frontend\n\
  --fortran             Has the Fortran frontend been built (ON or OFF)\n\
  --fortran-frontend    Path to the Fortran frontend\n\
  --help                Print this help message\n\
  --hip-prefix          The rocm installation used by the hip tapir target\n\
  --hip-target          Has the hip tapir target been built (ON or OFF)\n\
  --kitsune-version     Print the Kitsune version\n\
  --known-langs         All languages supported by Kitsune\n\
  --known-tapir-targets The names of all known tapir targets\n\
  --kokkos-mode         Has Kokkos mode been enabled (ON or OFF)\n\
  --lambda-target       Has the lambda tapir target been built (ON or OFF)\n\
  --langs               The languages for which a frontend has been built\n\
  --llvm-version        Print the LLVM version on which this is based\n\
  --omptask-target      Has the omptask tapir target been built (ON or OFF)\n\
  --opencilk-target     Has the opencilk tapir target been built (ON or OFF)\n\
  --openmp-target       Has the openmp tapir target been built (ON or OFF)\n\
  --prefix              Prefix for this Kitsune installation\n\
  --pthreads-target     Has the pthreads tapir target been built (ON or OFF)\n\
  --qthreads-target     Has the qthreads tapir target been built (ON or OFF)\n\
  --realm-target        Has the realm tapir target been built (ON or OFF)\n\
  --rtlib-shared        Path to the runtime's dynamic shared object\n\
  --rtlib-static        Path to the runtime's static archive\n\
  --serial-target       Has the serial tapir target been built (ON or OFF)\n\
  --tapir-targets       The tapir targets that have been built\n\
  --version             Prints both LLVM and Kitsune versions\n\
\n\
When querying paths to frontends and prefixes, no output will be printed if\n\
that frontend or tapir target has not been built\n";
  if (exitWithFailure)
    exit(1);
}

// Get the path to this executable.
static std::string getMainExecutable(const char *argv0) {
  // The second argument is nominally the name of the main function, but taking
  // the address of main is not permitted. So we put in the address of some
  // other function.
  return sys::fs::getMainExecutable(argv0, /*main=*/(void *)usage);
}

static SmallString<256> getBinDir(const char *argv0) {
  return sys::path::parent_path(getMainExecutable(argv0));
}

static SmallString<256> getPrefix(const char *argv0) {
  SmallString<256> bin = getBinDir(argv0);
  return sys::path::parent_path(bin);
}

static SmallString<256> getResourceDir(const char *argv0) {
  SmallString<256> path(getPrefix(argv0));
  sys::path::append(path, kitLibDirName());
  sys::path::append(path, "kitsune");
  sys::path::append(path, std::to_string(LLVM_VERSION_MAJOR));
  return path;
}

static SmallString<256> getResourceLibDir(const char *argv0) {
  SmallString<256> path = getResourceDir(argv0);
  sys::path::append(path, "lib");
  return path;
}

static void render(bool b) { outs() << (b ? "ON" : "OFF") << "\n"; }

static void render(const char *s) { outs() << s << "\n"; }

static void render(StringRef s) { outs() << s << "\n"; }

static void renderIf(bool cond, StringRef s) {
  if (cond)
    outs() << s << "\n";
}

static void renderVersions() {
  outs() << "LLVM version: " << PACKAGE_VERSION << "\n";
  outs() << "Kitsune version: " << kitPackageVersion() << "\n";
}

static void renderPathIf(bool cond, const char *argv0, StringRef base) {
  if (cond) {
    SmallString<256> path = getBinDir(argv0);
    sys::path::append(path, base);
    outs() << path << "\n";
  }
}

static void renderRuntimeSharedLib(const char *argv0) {
  StringRef pfx = kitRuntimeSharedLibPrefix();
  StringRef sfx = kitRuntimeSharedLibSuffix();
  SmallString<256> path = getResourceLibDir(argv0);
  sys::path::append(path, join_items("", pfx, kitRuntimeSharedLibName(), sfx));

  outs() << path << "\n";
}

static void renderRuntimeStaticLib(const char *argv0) {
  StringRef pfx = kitRuntimeStaticLibPrefix();
  StringRef sfx = kitRuntimeStaticLibSuffix();
  SmallString<256> path = getResourceLibDir(argv0);
  sys::path::append(path, join_items("", pfx, kitRuntimeStaticLibName(), sfx));

  outs() << path << "\n";
}

int main(int argc, char **argv) {
  if (argc == 1)
    usage();

  for (int i = 1; i < argc; ++i) {
    StringRef arg = argv[i];
    if (arg == "--help")
      usage(false);
    else if (arg == "--c")
      render(kitCEnabled());
    else if (arg == "--c-frontend")
      renderPathIf(kitCEnabled(), argv[0], kitCFrontend());
    else if (arg == "--cuda-prefix")
      renderIf(isEnabledTT(TTID::Cuda), kitCudaPrefix());
    else if (arg == "--cuda-target")
      render(isEnabledTT(TTID::Cuda));
    else if (arg == "--custom-target")
      render(isEnabledTT(TTID::Custom));
    else if (arg == "--cxx")
      render(kitCXXEnabled());
    else if (arg == "--cxx-frontend")
      renderPathIf(kitCXXEnabled(), argv[0], kitCXXFrontend());
    else if (arg == "--fortran")
      render(kitFortranEnabled());
    else if (arg == "--fortran-frontend")
      renderPathIf(kitFortranEnabled(), argv[0], kitFortranFrontend());
    else if (arg == "--hip-prefix")
      renderIf(isEnabledTT(TTID::Hip), kitHipPrefix());
    else if (arg == "--hip-target")
      render(isEnabledTT(TTID::Hip));
    else if (arg == "--kitsune-version")
      render(kitPackageVersion());
    else if (arg == "--kokkos")
      render(kitKokkosEnabled());
    else if (arg == "--known-langs")
      render(kitKnownLangs());
    else if (arg == "--known-tapir-targets")
      render(kitKnownTapirTargets());
    else if (arg == "--lambda-target")
      render(isEnabledTT(TTID::Lambda));
    else if (arg == "--langs")
      render(kitEnabledLangs());
    else if (arg == "--llvm-version")
      render(PACKAGE_VERSION);
    else if (arg == "--omptask-target")
      render(isEnabledTT(TTID::OMPTask));
    else if (arg == "--opencilk-target")
      render(isEnabledTT(TTID::OpenCilk));
    else if (arg == "--openmp-target")
      render(isEnabledTT(TTID::OpenMP));
    else if (arg == "--pthreads-target")
      render(isEnabledTT(TTID::Pthreads));
    else if (arg == "--prefix")
      render(getPrefix(argv[0]));
    else if (arg == "--qthreads-target")
      render(isEnabledTT(TTID::Qthreads));
    else if (arg == "--realm-target")
      render(isEnabledTT(TTID::Realm));
    else if (arg == "--rtlib-shared")
      renderRuntimeSharedLib(argv[0]);
    else if (arg == "--rtlib-static")
      renderRuntimeStaticLib(argv[0]);
    else if (arg == "--serial-target")
      render(isEnabledTT(TTID::Serial));
    else if (arg == "--tapir-targets")
      render(kitEnabledTapirTargets());
    else if (arg == "--version")
      renderVersions();
    else
      usage();
  }
  return 0;
}
