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

#include <hccl.h>
#include <hccl_types.h>

#include "habana_helpers/logging.h"
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
    auto status = hcclCommDestroy(*hccl_handle_);
    HABANA_ASSERT(
        status == hcclSuccess,
        "hcclCommDestroy returned with an error status=",
        status);
    hccl_handle_.reset();
  }
}

std::shared_ptr<hccl_integration::device_context> HcclCommunicator::
    getDeviceCtxt() {
  if (device_context_ == nullptr) {
    std::call_once(init_flag, [this] { this->Init(); });
    device_context_ = std::make_shared<hccl_integration::device_context>(0);
  }
  return device_context_;
}

synStreamHandle HcclCommunicator::getCommStream() {
  if (comm_stream_ == nullptr) {
    auto devctx = getDeviceCtxt();
    synStreamHandle collective_stream;
    devctx->acquire_collective_stream(&collective_stream);
    comm_stream_ = collective_stream;
  }
  return comm_stream_;
}

std::shared_ptr<hcclComm_t> HcclCommunicator::GetHcclHandle() {
  return hccl_handle_;
}

std::shared_ptr<HcclCommunicator> HcclCommunicator::Create(
    int rank,
    int size,
    std::function<void(hcclUniqueId*)> broadcastUniqueHCCLID_fn) {
  std::shared_ptr<HcclCommunicator> comm(
      new HcclCommunicator(
          HcclCommunicator::next_id_++, rank, size, broadcastUniqueHCCLID_fn),
      [](auto p) {
        std::lock_guard<std::mutex> lock(
            HcclCommunicator::communicator_map_mutext_);
        communicator_map_.erase(p->GetId());
        delete p;
      });
  std::lock_guard<std::mutex> lock(HcclCommunicator::communicator_map_mutext_);
  communicator_map_[comm->GetId()] = std::weak_ptr<HcclCommunicator>(comm);
  return comm;
};

std::shared_ptr<HcclCommunicator> HcclCommunicator::Get(int64_t id) {
  std::lock_guard<std::mutex> lock(HcclCommunicator::communicator_map_mutext_);

  if (communicator_map_.find(id) == communicator_map_.end()) {
    return nullptr;
  }

  return communicator_map_.at(id).lock();
}

int HcclCommunicator::Count() {
  return HcclCommunicator::next_id_;
}

HcclCommunicator::HcclCommunicator(
    int64_t id,
    int rank,
    int size,
    std::function<void(hcclUniqueId*)> broadcastUniqueHCCLID_fn)
    : id_(id),
      size_(size),
      rank_(rank),
      comm_stream_(nullptr),
      broadcastUniqueHCCLID_fn_(broadcastUniqueHCCLID_fn) {}

void HcclCommunicator::Init() {
  hcclUniqueId hccl_id = {{0}, 0};
  PT_LAZY_DEBUG("HcclCommunicator init. id = ", id_);
  if (rank_ == 0) {
    hcclResult_t result{hcclGetUniqueId(&hccl_id)};
    HABANA_ASSERT(hcclSuccess == result && "Get HCCL UniqueId Error");
  }

  broadcastUniqueHCCLID_fn_(&hccl_id);

  hcclComm_t new_comm;
  if (!GET_ENV_FLAG_NEW(PT_HPU_EMULATE_DISTRIBUTED)) {
    hcclResult_t result{hcclCommInitRank(&new_comm, size_, hccl_id, rank_)};
    HABANA_ASSERT(hcclSuccess == result && "Comm Init Rank Error");
    hccl_handle_ = std::make_shared<hcclComm_t>(new_comm);
  }
}

std::atomic_int64_t HcclCommunicator::next_id_ = 0;
std::unordered_map<int64_t, std::weak_ptr<HcclCommunicator>>
    HcclCommunicator::communicator_map_;
std::mutex HcclCommunicator::communicator_map_mutext_;

} // namespace habana
