/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include <gtest/gtest.h>
#include <cstring>
#include "utils/rerun_failures.h"
#include "habana_helpers/logging.h"

int main(int argc, char* argv[]) {
  bool reruns = false;
  for (int i = 0; i < argc; ++i) {
    if (std::strcmp(argv[i], "--rerun-fail") == 0) {
      reruns = true;
    }
  }
  ::testing::InitGoogleTest(&argc, argv);
  if (reruns) {
    CustomTestRunner runner = CustomTestRunner();
    auto listener = std::make_unique<RetryOnFailureListener>(runner);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    listeners.Append(listener.get());
    int result = runner.RunAllTests();
    int rerun_result = runner.RunFailedTests();
    if (rerun_result == 0) {
      result =
          0; // If rerun tests pass, consider the overall result as successful
    }
    listeners.Release(listener.get());
    return result;
  } else {
    try {
      return RUN_ALL_TESTS();
    } catch (...) {
      PT_TEST_DEBUG("Caught unknown exception");
      return 1;
    }
  }
}
