// RUN: %kitcc -E %s -o - | FileCheck --check-prefixes=ID,TARGET-NOT %s
// RUN: %kitcc -ftapir=serial -E %s -o - | FileCheck --check-prefixes=ID,TARGET %s

// The compile ID macros are also defined when just using clang. We may want to
// change this so they are only set when using the Kitsune frontend.
// RUN: %clang -E %s -o - | FileCheck --check-prefixes=ID,TARGET-NOT %s

// ID: return 1;
// ID-NOT: return 0;

int isKitsune() {
#ifdef __kitsune__
  return 1;
#else
  return 0;
#endif // __kitsune
}

// TARGET: return "serial";
// TARGET-NOT: return "";

const char *getTapirTarget() {
#ifdef __kitsune_tt__
  return __kitsune_tt__;
#else
  return "";
#endif // __kitsune_tt_
}
