/**
 * Copyright (c) 2025 Intel Corporation
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
#include "rerun_failures.h"
#include <iostream>

void CustomTestRunner::AddFailedTest(
    const std::string& test_case_name,
    const std::string& test_name) {
  failed_tests_.emplace_back(test_case_name + "." + test_name);
}

int CustomTestRunner::RunAllTests() {
  return RUN_ALL_TESTS();
}

int CustomTestRunner::RunFailedTests() {
  if (!failed_tests_.empty()) {
    std::string filter = "";
    for (const auto& test : failed_tests_) {
      if (!filter.empty()) {
        filter += ":";
      }
      filter += test;
    }
    std::cout << "Retrying failed tests with filter: " << filter << "\n";
    ::testing::GTEST_FLAG(filter) = filter;
    int rerun_result = RUN_ALL_TESTS();
    failed_tests_.clear();
    return rerun_result;
  }
  return 0; // No failed tests to rerun
}

void RetryOnFailureListener::OnTestEnd(const ::testing::TestInfo& test_info) {
  if (test_info.result()->Failed()) {
    runner_.AddFailedTest(test_info.test_case_name(), test_info.name());
  }
}