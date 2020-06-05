/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <iostream>
#include <ostream>
#include <string>
#define TEST(a, b)                   \
  void test();                       \
  int main(int argc, char* argv[]) { \
    test();                          \
    return 0;                        \
  }                                  \
  void test()

class t1000 {
 public:
  t1000(std::string where, int line, bool kill) : kill_(kill) {
    std::cerr << " line " << std::to_string(line) << " assertion " << where << " failed.\n";
  }

  ~t1000() {
    if (kill_) {
      std::cerr << std::flush;
      std::exit(1);
    }
  }

  bool kill_{};
};

template <typename T>
t1000&& operator<<(t1000&& t, T&& arg) {
  std::cerr << arg << std::flush;
  return std::move(t);
}

#define EXPECT_EQ(a, b) \
  if ((a) == (b)) {     \
  } else                \
    t1000 { #a " == " #b, __LINE__, false }

#define ASSERT_EQ(a, b) \
  if ((a) == (b)) {     \
  } else                \
    t1000 { #a " == " #b, __LINE__, true }
#define ASSERT_TRUE(a) ASSERT_EQ(true, a)
#define ASSERT_NE(a, b) \
  if ((a) != (b)) {     \
  } else                \
    t1000 { #a " != " #b, __LINE__, true }
