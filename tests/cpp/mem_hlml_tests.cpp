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

#include <fcntl.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <unistd.h>
#include <memory>
#include <stdexcept>
#include "backend/synapse_helpers/mem_hlml.h"

using namespace synapse_helpers;

struct MemHlMlReporterTests : public ::testing::Test {
  void SetUp() override {
    memory_reporter = std::make_shared<HlMlMemoryReporter>(15, false);
  }

  void TearDown() override {
    memory_reporter.reset();
  }

  hlml_shm_data ReadData() {
    int fd = ::open(Path(), O_RDWR);
    if (fd == -1) {
      throw std::runtime_error("cannot open file");
    }

    hlml_shm_data buffer;
    auto bytes = ::read(fd, &buffer, sizeof(buffer));
    ::close(fd);
    if (bytes != sizeof(buffer)) {
      throw std::runtime_error("File too small");
    }

    return buffer;
  }

  const char* Path() {
    return "/dev/shm" HLML_SHM_DEVICE_NAME_PREFIX "15";
  }

  void AssertTimestamp(std::uint64_t expected, int threshold = 2) {
    // Checks timestamp with error tolerance up to 1 second.
    auto actual = ReadData().timestamp;
    auto diff = expected - actual;
    if (actual > expected) {
      diff = actual - expected;
    }
    ASSERT_LT(diff, threshold);
  }

  std::shared_ptr<HlMlMemoryReporter> memory_reporter;
};

TEST_F(MemHlMlReporterTests, Creation) {
  int err = access(Path(), O_RDWR);
  ASSERT_EQ(err, 0);
  ASSERT_EQ(ReadData().version, HLML_SHM_VERSION);
}

TEST_F(MemHlMlReporterTests, CleanUp) {
  memory_reporter.reset();
  int err = access(Path(), O_RDWR);
  ASSERT_EQ(err, -1);
}

TEST_F(MemHlMlReporterTests, PublishMemory) {
  const std::uint64_t value1 = 0x1122334455667788UL;
  const std::uint64_t value2 = 0xaa22334455667788UL;
  auto old_timestamp = ReadData().timestamp;
  memory_reporter->PublishMemory(value1);
  ASSERT_EQ(ReadData().used_mem_in_bytes, value1);
  ASSERT_EQ(ReadData().timestamp, old_timestamp);
  memory_reporter->PublishMemory(value2);
  ASSERT_EQ(ReadData().used_mem_in_bytes, value2);
  ASSERT_EQ(ReadData().timestamp, old_timestamp);
}

TEST_F(MemHlMlReporterTests, PublishTimestamp) {
  auto old_value = ReadData().used_mem_in_bytes;
  memory_reporter->PublishTimestamp();
  ASSERT_EQ(ReadData().used_mem_in_bytes, old_value);
  AssertTimestamp(time(NULL));
  memory_reporter->PublishTimestamp();
  ASSERT_EQ(ReadData().used_mem_in_bytes, old_value);
  AssertTimestamp(time(NULL));
}

struct MemHlMlUpdaterTests : public MemHlMlReporterTests {
  void SetUp() override {
    MemHlMlReporterTests::SetUp();

    auto get_used_memory = [&] { return memory_value; };

    memory_updater =
        std::make_shared<HlMlMemoryUpdater>(memory_reporter, get_used_memory);
  }

  void TearDown() override {
    memory_updater.reset();
    MemHlMlReporterTests::TearDown();
  }

  void WaitForUpdate(int factor = 1) {
    sleep(factor * HlMlMemoryUpdater::INTERVAL);
  }

  volatile std::uint64_t memory_value = 0xaa00;
  std::shared_ptr<HlMlMemoryUpdater> memory_updater;
};

TEST_F(MemHlMlUpdaterTests, InitialValue) {
  WaitForUpdate();
  ASSERT_EQ(memory_value, ReadData().used_mem_in_bytes);
  AssertTimestamp(time(NULL));
}

TEST_F(MemHlMlUpdaterTests, UpdatingValueInBackground) {
  memory_value = 0xbbcc;
  WaitForUpdate(2);
  ASSERT_EQ(memory_value, ReadData().used_mem_in_bytes);
  AssertTimestamp(time(NULL), 3);
  memory_value = 0xddee;
  WaitForUpdate(2);
  AssertTimestamp(time(NULL), 3);
  ASSERT_EQ(memory_value, ReadData().used_mem_in_bytes);
}