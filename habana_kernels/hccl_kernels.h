/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include "habana_kernels/habana_operator.h"

namespace synapse_helpers {
using event_done_callback = std::function<void()>;
}

namespace habana {

class CollectiveOperator : public habana::HabanaOperator {
 public:
  CollectiveOperator() = delete;
  CollectiveOperator(const std::string guid) : HabanaOperator(guid){};
  virtual void RunCollective(
      std::vector<PtTensorInfoShared>& inputs,
      bool async,
      synapse_helpers::event_done_callback done_cb) = 0;
  const std::string& GetGuid() {
    return guid_;
  }
};

class HcclBroadcastOperator : public CollectiveOperator {
 public:
  HcclBroadcastOperator(int device_id, c10::ScalarType scalarType)
      : CollectiveOperator("hccl::broadcast_") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  // TODO: SW-68563 add patching function to be called when the recipe is
  // desiralized update internal mebers e.g device_, root_rank
  virtual void RunCollective(
      std::vector<PtTensorInfoShared>& inputs,
      bool async,
      synapse_helpers::event_done_callback done_cb);

 private:
  int64_t device_;
  int64_t comm_id_;
  int root_rank_;
  at::ScalarType data_type_;
};

class HcclAllreduceOperator : public CollectiveOperator {
 public:
  HcclAllreduceOperator(int device_id, c10::ScalarType scalarType)
      : CollectiveOperator("hccl::allreduce_") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual void RunCollective(
      std::vector<PtTensorInfoShared>& inputs,
      bool async,
      synapse_helpers::event_done_callback done_cb);
  int64_t device_;
  int64_t reduce_op_;
  int64_t comm_id_;
  at::ScalarType data_type_;
};

} // namespace habana