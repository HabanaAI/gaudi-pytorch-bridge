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

#include "mem_hlml.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

namespace synapse_helpers {

std::string HlMlMemoryReporter::MakePath(int device_index) {
  std::string path;
  path += HLML_SHM_DEVICE_NAME_PREFIX;
  path += std::to_string(device_index);
  return path;
}

HlMlMemoryReporter::HlMlMemoryReporter(int device_index) {
  m_path = MakePath(device_index);
  m_data = MmapSharedObject();
}

HlMlMemoryReporter::~HlMlMemoryReporter() {
  ::munmap(m_data, sizeof(*m_data));
  ::shm_unlink(m_path.c_str());
}

void HlMlMemoryReporter::PublishMemory(std::uint64_t bytes) {
  m_data->used_mem_in_bytes = bytes;
}

void HlMlMemoryReporter::PublishTimestamp() {
  m_data->timestamp = time(NULL);
}

int HlMlMemoryReporter::OpenSharedObject() {
  int flags = O_CREAT | O_TRUNC | O_RDWR;

  int fd = shm_open(m_path.c_str(), flags, 0777);
  if (fd == -1) {
    throw Error("shm_open", errno);
  }

  return fd;
}

hlml_shm_data* HlMlMemoryReporter::MmapSharedObject() {
  int fd = OpenSharedObject();

  try {
    auto data = PrepareSharedObject(fd);
    ::close(fd);
    return data;
  } catch (const Error&) {
    // If something went wrong just clean up resources and continue
    // exceptional flow.
    ::close(fd);
    ::shm_unlink(m_path.c_str());
    throw;
  }
}

hlml_shm_data* HlMlMemoryReporter::PrepareSharedObject(int fd) {
  int err = ::ftruncate(fd, sizeof(*m_data));
  if (err == -1) {
    throw Error("ftruncate", errno);
  }

  void* ptr =
      ::mmap(0, sizeof(*m_data), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == (void*)MAP_FAILED) {
    throw Error("mmap", errno);
  }

  auto* object = reinterpret_cast<hlml_shm_data*>(ptr);
  object->version = HLML_SHM_VERSION;
  return object;
}

namespace {
const char* FillErrorMessage(char* buffer, const char* operation, int error) {
  snprintf(
      buffer,
      HlMlMemoryReporter::Error::MAXLEN - 1,
      "mem_hlml failed: %s: errno %i: %s",
      operation,
      error,
      strerror(error));
  buffer[HlMlMemoryReporter::Error::MAXLEN - 1] = 0;
  return buffer;
}
} // namespace

HlMlMemoryReporter::Error::Error(const char* operation, int error)
    : runtime_error(FillErrorMessage(buffer, operation, error)) {}

HlMlMemoryUpdater::HlMlMemoryUpdater(
    std::shared_ptr<HlMlMemoryReporter> reporter,
    std::function<std::uint64_t()> get_used_memory)
    : m_reporter(reporter),
      m_get_used_memory(get_used_memory),
      m_thread([&] { thread_main(); }) {}

HlMlMemoryUpdater::~HlMlMemoryUpdater() {
  stop();
  m_thread.join();
}

void HlMlMemoryUpdater::stop() {
  m_quit.store(true);
}

void HlMlMemoryUpdater::thread_main() {
  while (not m_quit.load()) {
    m_reporter->PublishMemory(m_get_used_memory());
    m_reporter->PublishTimestamp();
    sleep(INTERVAL);
  }
}

} // namespace synapse_helpers