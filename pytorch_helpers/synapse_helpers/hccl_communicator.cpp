/******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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

#include <hccl.h>
#include <hccl_types.h>
#include <hcl_api.h>

#include "hccl_communicator.h"

namespace habana {

int64_t HcclCommunicator::GetId() const {
  return id_;
}

int64_t HcclCommunicator::GetRank() const {
  return rank_;
}

int64_t HcclCommunicator::GetSize() const {
  return size_;
}

HcclCommunicator::~HcclCommunicator() {
  PT_LAZY_DEBUG("HcclCommunicator destroy. id = ", id_);
  if (hccl_handle_) {
    hcclCommDestroy(*hccl_handle_);
    hccl_handle_.reset();
  }
}

std::shared_ptr<hccl_integration::device_context> HcclCommunicator::
    getDeviceCtxt(int deviceId) {
  if (device_contexts_.find(deviceId) == device_contexts_.end()) {
    std::call_once(init_flag, [this] { this->Init(); });
    device_contexts_[deviceId] =
        std::make_shared<hccl_integration::device_context>(deviceId);
  }
  return device_contexts_.at(deviceId);
}

std::vector<hcclStream_t> HcclCommunicator::getCommStreams() {
  std::vector<hcclStream_t> streams;
  for (auto const& [device_id, stream] : comm_streams_) {
    streams.emplace_back(stream);
  }
  return streams;
}

hcclStream_t HcclCommunicator::getCommStream(int deviceId) {
  if (comm_streams_.find(deviceId) == comm_streams_.end()) {
    auto devctx = getDeviceCtxt(deviceId);
    hcclStream_t collective_stream_;
    devctx->acquire_collective_stream(&collective_stream_);
    comm_streams_[deviceId] = collective_stream_;
  }
  return comm_streams_.at(deviceId);
}

std::shared_ptr<hcclComm_t> HcclCommunicator::GetHcclHandle() {
  return hccl_handle_;
}

std::shared_ptr<HcclCommunicator> HcclCommunicator::Create(
    int rank,
    int size,
    std::function<void(int64_t, hcclUniqueId*)> broadcastUniqueHCCLID_fn) {
  static std::atomic_int64_t next_id(0);

  std::shared_ptr<HcclCommunicator> comm(
      new HcclCommunicator(next_id++, rank, size, broadcastUniqueHCCLID_fn),
      [](auto p) {
        communicator_map_.erase(p->GetId());
        delete p;
      });
  communicator_map_[comm->GetId()] = std::weak_ptr<HcclCommunicator>(comm);
  return comm;
};

std::shared_ptr<HcclCommunicator> HcclCommunicator::Get(int64_t id) {
  return communicator_map_.at(id).lock();
}

HcclCommunicator::HcclCommunicator(
    int64_t id,
    int rank,
    int size,
    std::function<void(int64_t, hcclUniqueId*)> broadcastUniqueHCCLID_fn)
    : id_(id),
      size_(size),
      rank_(rank),
      broadcastUniqueHCCLID_fn_(broadcastUniqueHCCLID_fn) {}

void HcclCommunicator::Init() {
  hcclUniqueId hccl_id;
  PT_LAZY_DEBUG("HcclCommunicator init. id = ", id_);
  if (rank_ == 0) {
    hcclResult_t result{hcclGetUniqueId(&hccl_id)};
    HABANA_ASSERT(hcclSuccess == result && "Get HCCL UniqueId Error");
  }

  broadcastUniqueHCCLID_fn_(id_, &hccl_id);

  hcclComm_t new_comm;
  hcclResult_t result{hcclCommInitRank(&new_comm, size_, hccl_id, rank_)};
  HABANA_ASSERT(hcclSuccess == result && "Comm Init Rank Error");
  std::lock_guard<std::mutex> lock(mutex_);
  hccl_handle_ = std::make_shared<hcclComm_t>(new_comm);
}

std::unordered_map<int64_t, std::weak_ptr<HcclCommunicator>>
    HcclCommunicator::communicator_map_;

} // namespace habana
