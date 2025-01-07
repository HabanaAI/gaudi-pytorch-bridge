/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
