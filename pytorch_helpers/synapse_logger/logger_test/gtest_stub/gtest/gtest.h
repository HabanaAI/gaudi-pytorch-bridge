/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
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
    std::cerr << " line " << std::to_string(line) << " assertion " << where
              << " failed.\n";
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

#define EXPECT_EQ(a, b)             \
  if ((a) == (b)) {                 \
  } else                            \
    t1000 {                         \
#a " == " #b, __LINE__, false \
    }

#define ASSERT_EQ(a, b)            \
  if ((a) == (b)) {                \
  } else                           \
    t1000 {                        \
#a " == " #b, __LINE__, true \
    }
#define ASSERT_TRUE(a) ASSERT_EQ(true, a)
#define ASSERT_NE(a, b)            \
  if ((a) != (b)) {                \
  } else                           \
    t1000 {                        \
#a " != " #b, __LINE__, true \
    }
