/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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
#include "process_group_hccl_base.hpp"

#include <hccl.h>
#include <hccl_types.h>
#include <unistd.h>
#include <future>
#include <map>

#include <pt_ver/2.2/torch_ver/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include "backend/helpers/collective_utils.h"
#include "common/utils.h"
#include "habana_helpers/logging.h"

namespace c10d {
namespace ops {
using Work = c10d_ver::Work;

c10::intrusive_ptr<Work> send_hpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t dstRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::HPU)
      ->send(tensor_vec, static_cast<int>(dstRank), static_cast<int>(tag));
}

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("send", send_hpu_);
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

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("recv_", recv_hpu_);
}

c10::intrusive_ptr<Work> recv_any_source_hpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::HPU)
      ->recvAnysource(tensor_vec, static_cast<int>(tag));
}

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("recv_any_source_", recv_any_source_hpu_);
}

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
              *reduce_op.get(),
              root_rank,
              root_tensor,
              std::chrono::milliseconds(timeout)});
}

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("reduce_", reduce_hpu_);
}

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
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(tensor_vec), work);
}

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("broadcast_", broadcast_hpu_);
}

// Return input tensors as output tensors to make inplace allreduce look like
// a functional API, so that make_fx can correctly build the dependencies in
// the graph later.
std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> allreduce_hpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    [[maybe_unused]] const c10::optional<at::Tensor>& sparse_indices,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::HPU)
          ->allreduce(
              tensor_vec,
              AllreduceOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout)});
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(tensor_vec), work);
}

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("allreduce_", allreduce_hpu_);
}

c10::intrusive_ptr<Work> allreduce_coalesced_hpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  AllreduceCoalescedOptions opts = AllreduceCoalescedOptions{};
  opts.reduceOp = *reduce_op.get();
  opts.timeout = std::chrono::milliseconds(timeout);
  return process_group->getBackend(c10::DeviceType::HPU)
      ->allreduce_coalesced(tensor_vec, opts);
}

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("allreduce_coalesced_", allreduce_coalesced_hpu_);
}

// Copy output tensors (not storage) so that this can be used in a functional
// manner
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
  return std::
      tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>(
          output_tensors, work);
}

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("allgather_", allgather_hpu_);
}

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
  return std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(output_tensor, work);
}

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("_allgather_base_", _allgather_base_hpu_);
}

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

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("allgather_coalesced_", allgather_coalesced_hpu_);
}

c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced_hpu_(
    at::TensorList outputs,
    at::TensorList inputs,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group) {
  auto output_vec = outputs.vec();
  auto input_vec = inputs.vec();
  return process_group->getBackend(c10::DeviceType::HPU)
      ->allgather_into_tensor_coalesced(output_vec, input_vec);
}

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl(
      "allgather_into_tensor_coalesced_", allgather_into_tensor_coalesced_hpu_);
}

void startCoalescing_(
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group) {
  return process_group->getBackend(c10::DeviceType::HPU)->startCoalescing();
}

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("startCoalescing", startCoalescing_);
}

c10::intrusive_ptr<c10d::Work> endCoalescing_(
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group) {
  return process_group->getBackend(c10::DeviceType::HPU)->endCoalescing();
}

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("endCoalescing", endCoalescing_);
}

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
                  *reduce_op.get(), std::chrono::milliseconds(timeout)});
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      output_tensors_vec, work);
}

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("reduce_scatter_", reduce_scatter_hpu_);
}

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
                          *reduce_op.get(),
                          std::chrono::milliseconds(timeout),
                          asyncOp});
  return std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(output_tensor, work);
}

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("_reduce_scatter_base_", _reduce_scatter_base_hpu_);
}

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
              *reduce_op.get(), std::chrono::milliseconds(timeout)});
}

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl(
      "reduce_scatter_tensor_coalesced_", reduce_scatter_tensor_coalesced_hpu_);
}

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

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("gather_", gather_hpu_);
}

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
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(output_tensors_vec), work);
}

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("scatter_", scatter_hpu_);
}

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
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(output_tensors_vec), work);
}

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("alltoall_", alltoall_hpu_);
}

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

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("alltoall_base_", alltoall_base_hpu_);
}

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

TORCH_LIBRARY_IMPL(c10d, HPU, m) {
  m.impl("barrier", barrier_hpu_);
}

} // namespace ops

} // namespace c10d