/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

#include <atomic>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "device_context.h"

namespace habana {

class HcclCommunicator {
 public:
  HcclCommunicator() = delete;
  int64_t GetId() const;
  int64_t GetRank() const;
  int64_t GetSize() const;

  virtual ~HcclCommunicator();

  std::shared_ptr<hccl_integration::device_context> getDeviceCtxt();
  synStreamHandle getCommStream();
  std::shared_ptr<hcclComm_t> GetHcclHandle();

  void flush_stream() {
    if (device_context_) {
      device_context_->flush_stream_events();
    }
  }

  static std::shared_ptr<HcclCommunicator> Create(
      int rank,
      int size,
      std::function<void(hcclUniqueId*)> broadcastUniqueHCCLID_fn);
  static std::shared_ptr<HcclCommunicator> Get(int64_t id);
  static int Count();

 private:
  HcclCommunicator(
      int64_t id,
      int rank,
      int size,
      std::function<void(hcclUniqueId*)> broadcastUniqueHCCLID_fn);
  void Init();

  int64_t id_;
  int size_;
  int rank_;
  std::shared_ptr<hcclComm_t> hccl_handle_;
  std::shared_ptr<hccl_integration::device_context> device_context_;
  synStreamHandle comm_stream_;
  std::function<void(hcclUniqueId*)> broadcastUniqueHCCLID_fn_;
  std::once_flag init_flag;

  static std::mutex communicator_map_mutext_;
  static std::unordered_map<int64_t, std::weak_ptr<HcclCommunicator>>
      communicator_map_;
  static std::atomic_int64_t next_id_;
};

} // namespace habana
