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
#include "backend/habana_operator.h"

namespace synapse_helpers {
using event_done_callback = std::function<void()>;
}

namespace habana {

class CollectiveOperator : public habana::HabanaOperator {
 public:
  CollectiveOperator() = delete;
  CollectiveOperator(
      const std::string guid,
      int device_id,
      c10::ScalarType scalar_type)
      : HabanaOperator(guid),
        device_id_(device_id),
        scalar_type_(scalar_type){};
  virtual void RunCollective(
      const std::vector<PtTensorInfoShared>& inputs,
      bool async,
      synapse_helpers::event_done_callback done_cb) const = 0;

  int GetDeviceId() const {
    return device_id_;
  };
  c10::ScalarType GetScalarType() const {
    return scalar_type_;
  };

  virtual void Serialize(std::ostream& os) const = 0;
  virtual void Deserialize(std::istream& is) = 0;

 protected:
  int device_id_;
  c10::ScalarType scalar_type_;
};

class HcclBroadcastOperator : public CollectiveOperator {
 public:
  HcclBroadcastOperator(int device_id, c10::ScalarType scalar_type)
      : CollectiveOperator("hccl::broadcast_", device_id, scalar_type) {
    this->CreateSynContext(device_id);
  }
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  void Serialize(std::ostream& os) const override;
  void Deserialize(std::istream& is) override;

  void RunCollective(
      const std::vector<PtTensorInfoShared>& inputs,
      bool async,
      synapse_helpers::event_done_callback done_cb) const override;

 private:
  int64_t comm_id_;
  int root_rank_;
};

class HcclAllreduceOperator : public CollectiveOperator {
 public:
  HcclAllreduceOperator(int device_id, c10::ScalarType scalar_type)
      : CollectiveOperator("hccl::allreduce_", device_id, scalar_type) {
    this->CreateSynContext(device_id);
  }
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  void Serialize(std::ostream& os) const override;
  void Deserialize(std::istream& is) override;

  void RunCollective(
      const std::vector<PtTensorInfoShared>& inputs,
      bool async,
      synapse_helpers::event_done_callback done_cb) const override;

 private:
  uint8_t reduce_op_;
  int64_t comm_id_;
};

class HcclReduceOperator : public CollectiveOperator {
 public:
  HcclReduceOperator(int device_id, c10::ScalarType scalar_type)
      : CollectiveOperator("hccl::reduce_", device_id, scalar_type) {
    this->CreateSynContext(device_id);
  }
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  void Serialize(std::ostream& os) const override;
  void Deserialize(std::istream& is) override;

  void RunCollective(
      const std::vector<PtTensorInfoShared>& inputs,
      bool async,
      synapse_helpers::event_done_callback done_cb) const override;

 private:
  int64_t dst_rank_;
  uint8_t reduce_op_;
  int64_t comm_id_;
};

class HcclAllToAllOutOperator : public CollectiveOperator {
 public:
  HcclAllToAllOutOperator(int device_id, c10::ScalarType scalar_type)
      : CollectiveOperator("hccl::alltoall_out", device_id, scalar_type) {
    this->CreateSynContext(device_id);
  }
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  void Serialize(std::ostream& os) const override;
  void Deserialize(std::istream& is) override;

  void RunCollective(
      const std::vector<PtTensorInfoShared>& inputs,
      bool async,
      synapse_helpers::event_done_callback done_cb) const override;

 private:
  int64_t comm_id_;
  std::vector<int64_t> outputSplitSizes;
  std::vector<int64_t> inputSplitSizes;
};

class HcclAllgatherOutOperator : public CollectiveOperator {
 public:
  HcclAllgatherOutOperator(int device_id, c10::ScalarType scalar_type)
      : CollectiveOperator("hccl::allgather_out", device_id, scalar_type) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  void Serialize(std::ostream& os) const override;
  void Deserialize(std::istream& is) override;

  void RunCollective(
      const std::vector<PtTensorInfoShared>& inputs,
      bool async,
      synapse_helpers::event_done_callback done_cb) const override;

 private:
  int64_t comm_id_;
};

class HcclReduceScatterOutOperator : public CollectiveOperator {
 public:
  HcclReduceScatterOutOperator(int device_id, c10::ScalarType scalar_type)
      : CollectiveOperator("hccl::reduce_scatter_out", device_id, scalar_type) {
    this->CreateSynContext(device_id);
  }
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  void Serialize(std::ostream& os) const override;
  void Deserialize(std::istream& is) override;

  void RunCollective(
      const std::vector<PtTensorInfoShared>& inputs,
      bool async,
      synapse_helpers::event_done_callback done_cb) const override;

 private:
  uint8_t reduce_op_;
  int64_t comm_id_;
};
class HcclSendOperator : public CollectiveOperator {
 public:
  HcclSendOperator(int device_id, c10::ScalarType scalar_type)
      : CollectiveOperator("hccl::send_", device_id, scalar_type) {
    this->CreateSynContext(device_id);
  }
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  void Serialize(std::ostream& os) const override;
  void Deserialize(std::istream& is) override;

  void RunCollective(
      const std::vector<PtTensorInfoShared>& inputs,
      bool async,
      synapse_helpers::event_done_callback done_cb) const override;

 private:
  int64_t dst_rank_;
  int64_t tag_;
  int64_t comm_id_;
};
class HcclRecvOperator : public CollectiveOperator {
 public:
  HcclRecvOperator(int device_id, c10::ScalarType scalar_type)
      : CollectiveOperator("hccl::recv_", device_id, scalar_type) {
    this->CreateSynContext(device_id);
  }
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  void Serialize(std::ostream& os) const override;
  void Deserialize(std::istream& is) override;

  void RunCollective(
      const std::vector<PtTensorInfoShared>& inputs,
      bool async,
      synapse_helpers::event_done_callback done_cb) const override;

 private:
  int64_t src_rank_;
  int64_t tag_;
  int64_t comm_id_;
};
} // namespace habana