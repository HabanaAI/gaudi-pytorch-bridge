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
#include "process_group_hccl_base.hpp"

#include <hccl.h>
#include <hccl_types.h>
#include <unistd.h>
#include <future>
#include <map>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include "backend/helpers/collective_utils.h"
#include "common/utils.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/pt_version_check.h"

namespace c10d::ops {

c10::intrusive_ptr<Work> send_hpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t dstRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::HPU)
      ->send(tensor_vec, static_cast<int>(dstRank), static_cast<int>(tag));
}

c10::intrusive_ptr<Work> recv_hpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t srcRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::HPU)
      ->recv(tensor_vec, static_cast<int>(srcRank), static_cast<int>(tag));
}

c10::intrusive_ptr<Work> recv_any_source_hpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::HPU)
      ->recvAnysource(tensor_vec, static_cast<int>(tag));
}

#if IS_PYTORCH_AT_LEAST(2, 8)
c10::intrusive_ptr<Work> reduce_hpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t root_rank,
    int64_t root_tensor,
    bool async_op,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::HPU)
      ->reduce(
          tensor_vec,
          ReduceOptions{
              *reduce_op,
              root_rank,
              root_tensor,
              std::chrono::milliseconds(timeout),
              async_op});
}
#else
c10::intrusive_ptr<Work> reduce_hpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::HPU)
      ->reduce(
          tensor_vec,
          ReduceOptions{
              *reduce_op,
              root_rank,
              root_tensor,
              std::chrono::milliseconds(timeout)});
}
#endif

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> broadcast_hpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t root_tensor,
    bool asyncOp,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work = process_group->getBackend(c10::DeviceType::HPU)
                  ->broadcast(
                      tensor_vec,
                      BroadcastOptions{
                          root_rank,
                          root_tensor,
                          std::chrono::milliseconds(timeout),
                          asyncOp});
  return {std::move(tensor_vec), work};
}

// Return input tensors as output tensors to make inplace allreduce look like
// a functional API, so that make_fx can correctly build the dependencies in
// the graph later.
#if IS_PYTORCH_AT_LEAST(2, 8)
std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> allreduce_hpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    [[maybe_unused]] const std::optional<at::Tensor>& sparse_indices,
    bool async_op,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work = process_group->getBackend(c10::DeviceType::HPU)
                  ->allreduce(
                      tensor_vec,
                      AllreduceOptions{
                          *reduce_op,
                          std::chrono::milliseconds(timeout),
                          async_op});
  return {std::move(tensor_vec), work};
}
#else
std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> allreduce_hpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    [[maybe_unused]] const std::optional<at::Tensor>& sparse_indices,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::HPU)
          ->allreduce(
              tensor_vec,
              AllreduceOptions{
                  *reduce_op, std::chrono::milliseconds(timeout)});
  return {std::move(tensor_vec), work};
}
#endif

#if IS_PYTORCH_AT_LEAST(2, 8)
c10::intrusive_ptr<Work> allreduce_coalesced_hpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    bool async_op,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  AllreduceCoalescedOptions opts = AllreduceCoalescedOptions{};
  opts.reduceOp = *reduce_op;
  opts.timeout = std::chrono::milliseconds(timeout);
  opts.asyncOp = async_op;
  return process_group->getBackend(c10::DeviceType::HPU)
      ->allreduce_coalesced(tensor_vec, opts);
}
#else
c10::intrusive_ptr<Work> allreduce_coalesced_hpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  AllreduceCoalescedOptions opts = AllreduceCoalescedOptions{};
  opts.reduceOp = *reduce_op;
  opts.timeout = std::chrono::milliseconds(timeout);
  return process_group->getBackend(c10::DeviceType::HPU)
      ->allreduce_coalesced(tensor_vec, opts);
}
#endif

// Copy output tensors (not storage) so that this can be used in a functional
// manner
#if IS_PYTORCH_AT_LEAST(2, 8)
std::tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>
allgather_hpu_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    at::TensorList input_tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    bool async_op,
    int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::HPU)
          ->allgather(
              const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
              input_tensors_vec,
              AllgatherOptions{std::chrono::milliseconds(timeout), async_op});
  return {output_tensors, work};
}
#else
std::tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>
allgather_hpu_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    at::TensorList input_tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::HPU)
          ->allgather(
              const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
              input_tensors_vec,
              AllgatherOptions{std::chrono::milliseconds(timeout)});
  return {output_tensors, work};
}
#endif

std::tuple<at::Tensor, c10::intrusive_ptr<Work>> _allgather_base_hpu_(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    bool asyncOp,
    int64_t timeout) {
  auto work =
      process_group->getBackend(c10::DeviceType::HPU)
          ->_allgather_base(
              output_tensor,
              input_tensor,
              AllgatherOptions{std::chrono::milliseconds(timeout), asyncOp});
  return {output_tensor, work};
}

#if IS_PYTORCH_AT_LEAST(2, 8)
c10::intrusive_ptr<Work> allgather_coalesced_hpu_(
    const std::vector<std::vector<at::Tensor>>& output_lists,
    const at::TensorList& input_list,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    bool async_op) {
  auto input_list_vec = input_list.vec();
  AllgatherOptions opts = AllgatherOptions{};
  opts.asyncOp = async_op;
  return process_group->getBackend(c10::DeviceType::HPU)
      ->allgather_coalesced(
          const_cast<std::vector<std::vector<at::Tensor>>&>(output_lists),
          input_list_vec,
          opts);
}
#else
c10::intrusive_ptr<Work> allgather_coalesced_hpu_(
    const std::vector<std::vector<at::Tensor>>& output_lists,
    const at::TensorList& input_list,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group) {
  auto input_list_vec = input_list.vec();
  return process_group->getBackend(c10::DeviceType::HPU)
      ->allgather_coalesced(
          const_cast<std::vector<std::vector<at::Tensor>>&>(output_lists),
          input_list_vec);
}
#endif

#if IS_PYTORCH_AT_LEAST(2, 8)
c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced_hpu_(
    at::TensorList outputs,
    at::TensorList inputs,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    bool async_op) {
  auto output_vec = outputs.vec();
  auto input_vec = inputs.vec();
  AllgatherOptions opts = AllgatherOptions{};
  opts.asyncOp = async_op;
  return process_group->getBackend(c10::DeviceType::HPU)
      ->allgather_into_tensor_coalesced(output_vec, input_vec, opts);
}
#else
c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced_hpu_(
    at::TensorList outputs,
    at::TensorList inputs,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group) {
  auto output_vec = outputs.vec();
  auto input_vec = inputs.vec();
  return process_group->getBackend(c10::DeviceType::HPU)
      ->allgather_into_tensor_coalesced(output_vec, input_vec);
}
#endif

bool supportsCoalescing_([
    [maybe_unused]] const c10::intrusive_ptr<c10d::ProcessGroup>&
                             process_group) {
  return process_group->getBackend(c10::DeviceType::HPU)->supportsCoalescing();
}

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("supportsCoalescing", supportsCoalescing_);
}

void startCoalescing_(
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group) {
  process_group->getBackend(c10::DeviceType::HPU)->startCoalescing();
}

c10::intrusive_ptr<c10d::Work> endCoalescing_(
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group) {
  return process_group->getBackend(c10::DeviceType::HPU)->endCoalescing();
}

#if IS_PYTORCH_AT_LEAST(2, 8)
std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>
reduce_scatter_hpu_(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    bool async_op,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::HPU)
          ->reduce_scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ReduceScatterOptions{
                  *reduce_op,
                  std::chrono::milliseconds(timeout),
                  async_op});
  return {output_tensors_vec, work};
}
#else
std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>
reduce_scatter_hpu_(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::HPU)
          ->reduce_scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ReduceScatterOptions{
                  *reduce_op, std::chrono::milliseconds(timeout)});
  return {output_tensors_vec, work};
}
#endif

std::tuple<at::Tensor, c10::intrusive_ptr<Work>> _reduce_scatter_base_hpu_(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    bool asyncOp,
    int64_t timeout) {
  auto work = process_group->getBackend(c10::DeviceType::HPU)
                  ->_reduce_scatter_base(
                      output_tensor,
                      input_tensor,
                      ReduceScatterOptions{
                          *reduce_op,
                          std::chrono::milliseconds(timeout),
                          asyncOp});
  return {output_tensor, work};
}

#if IS_PYTORCH_AT_LEAST(2, 8)
c10::intrusive_ptr<c10d::Work> reduce_scatter_tensor_coalesced_hpu_(
    at::TensorList outputs,
    at::TensorList inputs,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    bool async_op,
    int64_t timeout) {
  auto output_vec = outputs.vec();
  auto input_vec = inputs.vec();
  return process_group->getBackend(c10::DeviceType::HPU)
      ->reduce_scatter_tensor_coalesced(
          output_vec,
          input_vec,
          ReduceScatterOptions{
              *reduce_op, std::chrono::milliseconds(timeout), async_op});
}
#else
c10::intrusive_ptr<c10d::Work> reduce_scatter_tensor_coalesced_hpu_(
    at::TensorList outputs,
    at::TensorList inputs,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto output_vec = outputs.vec();
  auto input_vec = inputs.vec();
  return process_group->getBackend(c10::DeviceType::HPU)
      ->reduce_scatter_tensor_coalesced(
          output_vec,
          input_vec,
          ReduceScatterOptions{
              *reduce_op, std::chrono::milliseconds(timeout)});
}
#endif

#if IS_PYTORCH_AT_LEAST(2, 8)
c10::intrusive_ptr<Work> gather_hpu_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const at::TensorList& input_tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t root_rank,
    bool async_op,
    int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  return process_group->getBackend(c10::DeviceType::HPU)
      ->gather(
          const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
          input_tensors_vec,
          GatherOptions{
              root_rank, std::chrono::milliseconds(timeout), async_op});
}
#else
c10::intrusive_ptr<Work> gather_hpu_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const at::TensorList& input_tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  return process_group->getBackend(c10::DeviceType::HPU)
      ->gather(
          const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
          input_tensors_vec,
          GatherOptions{root_rank, std::chrono::milliseconds(timeout)});
}
#endif

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> scatter_hpu_(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t root_rank,
    bool asyncOp,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::HPU)
          ->scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ScatterOptions{
                  root_rank, std::chrono::milliseconds(timeout), asyncOp});
  return {std::move(output_tensors_vec), work};
}

#if IS_PYTORCH_AT_LEAST(2, 8)
std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> alltoall_hpu_(
    const at::TensorList& output_tensors,
    const at::TensorList& input_tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    bool async_op,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto input_tensors_vec = input_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::HPU)
          ->alltoall(
              output_tensors_vec,
              input_tensors_vec,
              AllToAllOptions{std::chrono::milliseconds(timeout), async_op});
  return {std::move(output_tensors_vec), work};
}
#else
std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> alltoall_hpu_(
    const at::TensorList& output_tensors,
    const at::TensorList& input_tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto input_tensors_vec = input_tensors.vec();
  auto work = process_group->getBackend(c10::DeviceType::HPU)
                  ->alltoall(
                      output_tensors_vec,
                      input_tensors_vec,
                      AllToAllOptions{std::chrono::milliseconds(timeout)});
  return {std::move(output_tensors_vec), work};
}
#endif

#if IS_PYTORCH_AT_LEAST(2, 8)
c10::intrusive_ptr<Work> alltoall_base_hpu_(
    at::Tensor& output,
    at::Tensor& input,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    std::vector<int64_t> output_split_sizes,
    std::vector<int64_t> input_split_sizes,
    bool async_op,
    int64_t timeout) {
  return process_group->getBackend(c10::DeviceType::HPU)
      ->alltoall_base(
          output,
          input,
          output_split_sizes,
          input_split_sizes,
          AllToAllOptions{std::chrono::milliseconds(timeout), async_op});
}
#else
c10::intrusive_ptr<Work> alltoall_base_hpu_(
    at::Tensor& output,
    at::Tensor& input,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    std::vector<int64_t> output_split_sizes,
    std::vector<int64_t> input_split_sizes,
    int64_t timeout) {
  return process_group->getBackend(c10::DeviceType::HPU)
      ->alltoall_base(
          output,
          input,
          output_split_sizes,
          input_split_sizes,
          AllToAllOptions{std::chrono::milliseconds(timeout)});
}
#endif

#if IS_PYTORCH_AT_LEAST(2, 8)
c10::intrusive_ptr<Work> barrier_hpu_(
    at::Tensor /* unused */,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const std::vector<int64_t>& device_ids,
    bool async_op,
    int64_t timeout) {
  ::c10d::BarrierOptions opts;
  opts.device_ids = device_ids;
  opts.timeout = std::chrono::milliseconds(timeout);
  opts.asyncOp = async_op;
  return process_group->getBackend(c10::DeviceType::HPU)->barrier(opts);
}
#else
c10::intrusive_ptr<Work> barrier_hpu_(
    at::Tensor /* unused */,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const std::vector<int64_t>& device_ids,
    int64_t timeout) {
  ::c10d::BarrierOptions opts;
  opts.device_ids = device_ids;
  opts.timeout = std::chrono::milliseconds(timeout);
  return process_group->getBackend(c10::DeviceType::HPU)->barrier(opts);
}
#endif

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("_allgather_base_", _allgather_base_hpu_);
  m.impl("_reduce_scatter_base_", _reduce_scatter_base_hpu_);
  m.impl("allgather_", allgather_hpu_);
  m.impl("allgather_coalesced_", allgather_coalesced_hpu_);
  m.impl(
      "allgather_into_tensor_coalesced_", allgather_into_tensor_coalesced_hpu_);
  m.impl("allreduce_", allreduce_hpu_);
  m.impl("allreduce_coalesced_", allreduce_coalesced_hpu_);
  m.impl("alltoall_", alltoall_hpu_);
  m.impl("alltoall_base_", alltoall_base_hpu_);
  m.impl("barrier", barrier_hpu_);
  m.impl("broadcast_", broadcast_hpu_);
  m.impl("endCoalescing", endCoalescing_);
  m.impl("gather_", gather_hpu_);
  m.impl("recv_", recv_hpu_);
  m.impl("recv_any_source_", recv_any_source_hpu_);
  m.impl("reduce_", reduce_hpu_);
  m.impl("reduce_scatter_", reduce_scatter_hpu_);
  m.impl(
      "reduce_scatter_tensor_coalesced_", reduce_scatter_tensor_coalesced_hpu_);
  m.impl("scatter_", scatter_hpu_);
  m.impl("send", send_hpu_);
  m.impl("startCoalescing", startCoalescing_);
}

} // namespace c10d::ops
