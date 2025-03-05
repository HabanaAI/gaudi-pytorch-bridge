/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#ifndef GENERIC_RESOURCE_HOLDER_H
#define GENERIC_RESOURCE_HOLDER_H

#include <ATen/ATen.h>
#include <memory>
#include <vector>
#include "backend/habana_device/HPUStream.h"
#include "backend/synapse_helpers/device.h"
#include "backend/synapse_helpers/graph.h"

namespace synapse_helpers {
class device_ptr_lock;
}

class GenericResourceHolder {
 public:
  GenericResourceHolder() : stream_(c10::hpu::getCurrentHPUStream().unwrap()) {}

  explicit GenericResourceHolder(const at::Tensor& tensor)
      : tensors_{tensor}, stream_(c10::hpu::getCurrentHPUStream().unwrap()) {}

  GenericResourceHolder(
      const at::Tensor& tensor,
      bool non_blocking,
      c10::hpu::HPUStream stream,
      void* host_ptr)
      : tensors_{tensor},
        non_blocking_(non_blocking),
        stream_(stream),
        host_ptr_(host_ptr) {}

  GenericResourceHolder(
      std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe_id,
      synapse_helpers::active_recipe_counter* recipe_counter_ptr,
      size_t active_graph_key,
      const std::vector<at::Tensor>& input_tensors,
      const std::vector<at::Tensor>& output_tensors)
      : tensors_(),
        pt_tensors_(),
        input_tensors_(input_tensors),
        output_tensors_(output_tensors),
        recipe_id_(recipe_id),
        recipe_counter_ptr_(recipe_counter_ptr),
        active_graph_key_(active_graph_key),
        address_lock_(nullptr),
        non_blocking_(false),
        stream_(c10::hpu::getCurrentHPUStream().unwrap()),
        host_ptr_(nullptr) {}

  GenericResourceHolder(
      std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe_id,
      synapse_helpers::active_recipe_counter* recipe_counter_ptr,
      size_t active_graph_key)
      : tensors_(),
        pt_tensors_(),
        input_tensors_(),
        output_tensors_(),
        recipe_id_(recipe_id),
        recipe_counter_ptr_(recipe_counter_ptr),
        active_graph_key_(active_graph_key),
        address_lock_(nullptr),
        non_blocking_(false),
        stream_(c10::hpu::getCurrentHPUStream().unwrap()),
        host_ptr_(nullptr) {}

  GenericResourceHolder(const at::Tensor& tensor1, const at::Tensor& tensor2)
      : tensors_{tensor1},
        pt_tensors_{},
        input_tensors_{tensor1},
        output_tensors_{tensor2},
        recipe_id_(nullptr),
        recipe_counter_ptr_(nullptr),
        active_graph_key_(0),
        address_lock_(nullptr),
        non_blocking_(false),
        stream_(c10::hpu::getCurrentHPUStream().unwrap()),
        host_ptr_(nullptr) {}

  GenericResourceHolder(
      const at::Tensor& src,
      const at::Tensor& dst,
      bool non_blocking,
      c10::hpu::HPUStream stream,
      void* host_ptr)
      : src_(src),
        dst_(dst),
        tensors_{src, dst},
        non_blocking_(non_blocking),
        stream_(stream),
        host_ptr_(host_ptr) {}

  void add_tensor(const at::Tensor& tensor) {
    tensors_.push_back(tensor);
  }

  void add_pt_tensor(const at::Tensor& pt_tensor) {
    pt_tensors_.push_back(pt_tensor);
  }

  const std::vector<at::Tensor>& tensors() const {
    return tensors_;
  }

  const std::vector<at::Tensor>& pt_tensors() const {
    return pt_tensors_;
  }

  const std::vector<at::Tensor>& input_tensors() const {
    return input_tensors_;
  }

  const std::vector<at::Tensor>& output_tensors() const {
    return output_tensors_;
  }

  std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe_id() const {
    return recipe_id_;
  }

  synRecipeHandle get_syn_recipe_handle() const {
    return recipe_id_ ? recipe_id_->syn_recipe_handle_ : nullptr;
  }

  synapse_helpers::active_recipe_counter* recipe_counter_ptr() const {
    return recipe_counter_ptr_;
  }

  void increase_recipe_count() {
    if (recipe_counter_ptr_) {
      recipe_counter_ptr_->increase();
    }
  }

  void decrease_and_notify_recipe_count() {
    if (recipe_counter_ptr_) {
      recipe_counter_ptr_->decrease_and_notify();
    }
  }

  size_t active_graph_key() const {
    return active_graph_key_;
  }

  std::unique_ptr<synapse_helpers::device_ptr_lock>& get_address_lock() {
    return address_lock_;
  }

  void set_recipe_id(
      std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe) {
    recipe_id_ = std::move(recipe);
  }

  void set_input_tensors(const std::vector<at::Tensor>& tensors) {
    input_tensors_ = tensors;
  }

  void set_output_tensors(const std::vector<at::Tensor>& tensors) {
    output_tensors_ = tensors;
  }

  void set_active_graph_key(size_t key) {
    active_graph_key_ = key;
  }

  void set_recipe_counter_ptr(
      synapse_helpers::active_recipe_counter* counter_ptr) {
    recipe_counter_ptr_ = counter_ptr;
  }

  void release_resources() {
    tensors_.clear();
    pt_tensors_.clear();
    input_tensors_.clear();
    output_tensors_.clear();
    src_ = at::Tensor();
    dst_ = at::Tensor();
    recipe_id_.reset();
    recipe_counter_ptr_ = nullptr;
  }

  bool non_blocking() const {
    return non_blocking_;
  }

  c10::hpu::HPUStream stream() const {
    return c10::hpu::HPUStream(stream_);
  }

  void* host_ptr() const {
    return host_ptr_;
  }

  const at::Tensor& src() const {
    return src_;
  }

  const at::Tensor& dst() const {
    return dst_;
  }

 private:
  at::Tensor src_;
  at::Tensor dst_;
  std::vector<at::Tensor>
      tensors_; // General tensor storage for ensuring tensor lifetime.
  std::vector<at::Tensor> pt_tensors_; // PyTorch-specific tensors, used in
                                       // PyTorch-related operations.
  std::vector<at::Tensor>
      input_tensors_; // Input tensors for collective operations.
  std::vector<at::Tensor>
      output_tensors_; // Output tensors for collective operations.

  std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe_id_;
  synapse_helpers::active_recipe_counter* recipe_counter_ptr_ = nullptr;
  size_t active_graph_key_ = 0;

  std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock_;
  bool non_blocking_ = false;
  c10::Stream stream_;
  void* host_ptr_ = nullptr;
};

using GenericResourceHolderPtr = std::shared_ptr<GenericResourceHolder>;

#endif // GENERIC_RESOURCE_HOLDER_H
