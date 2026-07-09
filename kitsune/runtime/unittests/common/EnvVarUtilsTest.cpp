//===- envTest.cpp - Tests for environment variable parsing utilities -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common/env.h"

#include <limits>

#include "gtest/gtest.h"

namespace {

template <typename T> static T min() { return std::numeric_limits<T>::min(); }
template <typename T> static T max() { return std::numeric_limits<T>::max(); }

template <typename T> class VarSpec {
private:
  std::string _name;
  std::string _val;
  bool _valid;
  T _expected;

private:
public:
  VarSpec(std::string name, std::string val)
      : _name(name), _val(val), _valid(false) {}

  VarSpec(std::string name, std::string val, T expected)
      : _name(name), _val(val), _valid(true), _expected(expected) {}

  const char *name() const { return _name.c_str(); }
  const char *val() const { return _val.c_str(); }
  bool valid() const { return _valid; }
  T expected() const { return _expected; }
};

template <typename T> class EnvBase : public ::testing::Test {
private:
  std::vector<VarSpec<T>> vars;

private:
  std::string getName() {
    return std::string("TEST_") + std::to_string(vars.size());
  }

protected:
  void SetUp() override {
    for (const VarSpec<T> &v : vars)
      setenv(v.name(), v.val(), 1);
  }

  void TearDown() override {
    for (const VarSpec<T> &v : vars)
      unsetenv(v.name());
  }

  void add(const char *val) { vars.emplace_back(getName(), val); }

  void add(const char *val, T expected) {
    vars.emplace_back(getName(), val, expected);
  }

  void checkLookup() {
    for (const VarSpec<T> &v : vars) {
      T val;
      EXPECT_FALSE(__kitrt_env_lookup("NONEXISTENT", val));
      if (v.valid()) {
        EXPECT_TRUE(__kitrt_env_lookup(v.name(), val));
        EXPECT_EQ(val, v.expected());
      } else {
        EXPECT_FALSE(__kitrt_env_lookup(v.name(), val));
      }
    }
  }

  void checkSet(T expected) {
    T val;
    EXPECT_STREQ(getenv("TEST_0"), "");
    __kitrt_env_set("TEST_0", expected);
    EXPECT_TRUE(__kitrt_env_lookup("TEST_0", val));
    EXPECT_EQ(val, expected);
  }

  void checkUnset() {
    T val;
    __kitrt_env_unset("TEST_0");
    EXPECT_FALSE(getenv("TEST_0"));
    EXPECT_FALSE(__kitrt_env_lookup("TEST_0", val));
  }
};

class EnvBool : public EnvBase<bool> {
public:
  EnvBool() {
    // If the environment variable is present in the environment, even if its
    // value is not "truth-y", it will be treated as true.
    add("", true);
    add("1", true);
    add("T", true);
    add("true", true);
    add("TRUE", true);
    add("0", false);
    add("F", false);
    add("false", false);
    add("FALSE", false);
    add("s$@K$J8", true);
  }
};
TEST_F(EnvBool, __kitrt_env_lookup_bool) {
  checkLookup();
  checkSet(false);
  checkUnset();
}

class EnvI32 : public EnvBase<int> {
public:
  EnvI32() {
    add("");
    add("2147483648");
    add("-2147483649");
    add("2147483647", max<int32_t>());
    add("-2147483648", min<int32_t>());
    add("0", 0);
    add("1", 1);
    add("-1", -1);
    add("3.14159");
  }
};
TEST_F(EnvI32, __kitrt_env_lookup_i32) {
  checkLookup();
  checkSet(0xbadc0de);
  checkUnset();
}

class EnvU32 : public EnvBase<unsigned> {
public:
  EnvU32() {
    add("");
    add("4294967296");
    add("-4294967297");
    add("4294967295", max<uint32_t>());
    add("0", 0U);
    add("1", 1U);
    add("-1");
    add("3.14159");
  }
};
TEST_F(EnvU32, __kitrt_env_lookup_u32) {
  checkLookup();
  checkSet(0xbadc0de);
  checkUnset();
}

class EnvI64 : public EnvBase<int64_t> {
public:
  EnvI64() {
    add("");
    add("9223372036854775808");
    add("-9223372036854775809");
    add("9223372036854775807", max<int64_t>());
    add("-9223372036854775808", min<int64_t>());
    add("0", 0);
    add("1", 1);
    add("-1", -1);
    add("3.14159");
  }
};
TEST_F(EnvI64, __kitrt_env_lookup_i64) {
  checkLookup();
  checkSet(0xbadc0dedec0deL);
  checkUnset();
}

class EnvU64 : public EnvBase<uint64_t> {
public:
  EnvU64() {
    add("");
    add("18446744073709551616");
    add("-18446744073709551617");
    add("18446744073709551615", max<uint64_t>());
    add("0", 0);
    add("1", 1);
    add("-1", -1);
    add("3.14159");
  }
};
TEST_F(EnvI64, __kitrt_env_lookup_u64) {
  checkLookup();
  checkSet(0xbadc0dedec0deL);
  checkUnset();
}

class EnvFloat : public EnvBase<float> {
public:
  EnvFloat() {
    add("");
    add("0", 0.0f);
    add("1.5", 1.5f);
    add("-3.4", -3.4f);
    add("3.14E+02", 314.0f);
    add("3.14E+ab");
  }
};
TEST_F(EnvFloat, __kitrt_env_lookup_f32) {
  checkLookup();
  checkSet(2.71828f);
  checkUnset();
}

class EnvDouble : public EnvBase<double> {
public:
  EnvDouble() {
    add("");
    add("0", 0.0);
    add("1.5", 1.5);
    add("-3.4", -3.4);
    add("3.14E+02", 314.0);
    add("3.14E+ab");
  }
};
TEST_F(EnvDouble, __kitrt_env_lookup_f64) {
  checkLookup();
  checkSet(2.71828);
  checkUnset();
}

} // namespace
