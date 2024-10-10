/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include <hlml_shm.h>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace synapse_helpers {

/**
 * Memory reporter for HLML infrastructure. It is de facto communication
 * mechanism between framework and hl-smi command.
 *
 * The purpose of communication is to allow framework send precise information
 * about memory usage.
 */
class HlMlMemoryReporter {
 public:
  class Error : public std::runtime_error {
   public:
    static constexpr std::size_t MAXLEN = 256;
    Error(const char* operation, int error);

   private:
    char buffer[MAXLEN];
  };

  /**
   * Creates memory reporter for shared memory object associated with
   * given device.
   *
   * @param resolve_index - flag controlling if given device index should be
   *  corrected using synGetDeviceInfo. Useful only for testing.
   */
  HlMlMemoryReporter(int device_index, bool resolve_index = true);

  /**
   * During destruction we cleanup resources.
   */
  ~HlMlMemoryReporter();

  /**
   * Publish memory usage into shared object
   */
  void PublishMemory(std::uint64_t bytes);

  /**
   * Updates timestamp in shared object
   */
  void PublishTimestamp();

  /**
   * Get path
   */
  std::string GetPath() const;

 private:
  std::string MakePath(int device_index);
  int OpenSharedObject();
  hlml_shm_data* PrepareSharedObject(int fd);
  hlml_shm_data* MmapSharedObject();

  std::string m_path;
  hlml_shm_data* m_data = nullptr;
  int m_fd = -1;
};

/**
 * Background thread to update shared object with actual
 * value how much memory is used.
 */
class HlMlMemoryUpdater {
 public:
  static constexpr int INTERVAL = 1;
  HlMlMemoryUpdater(
      std::shared_ptr<HlMlMemoryReporter> reporter,
      std::function<std::uint64_t()> get_used_memory);
  ~HlMlMemoryUpdater();

  void stop();

 private:
  void thread_main();

  std::atomic_bool m_quit{false};
  std::shared_ptr<HlMlMemoryReporter> m_reporter;
  std::function<std::uint64_t()> m_get_used_memory;
  std::thread m_thread;
};

} // namespace synapse_helpers
