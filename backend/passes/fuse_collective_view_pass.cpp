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
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <typeinfo>
#include <unordered_map>

#include <ATen/record_function.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include <torch/csrc/api/include/torch/version.h>

#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/control_edges_processing.h"
#include "backend/passes/fuse_collective_view_pass.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/kernel_utils.h"

#include "backend/habana_device/tensor_builder.h"
#include "backend/helpers/tensor_info.h"
#include "backend/jitgraph_utils.h"
#include "habana_helpers/misc_utils.h"

using namespace torch::jit;
using namespace jitgraph_utils;
using namespace habana;

static std::tuple<int, int> GetInputOutputIndices(torch::jit::Node* node) {
  if (strcmp(node->kind().toQualString(), "hccl::alltoall_out") == 0) {
    return std::make_tuple(0, 4);
  } else if (strcmp(node->kind().toQualString(), "hccl::allgather_out") == 0) {
    return std::make_tuple(0, 2);
  }
  return std::make_tuple(-1, -1);
}

torch::jit::Value* FuseCollectiveViewPass::GetInputValue(
    torch::jit::Node* node,
    bool is_node_output) {
  auto indices = GetInputOutputIndices(node);
  int index;
  if (is_node_output) {
    index = std::get<1>(indices);
  } else {
    index = std::get<0>(indices);
  }
  if (index >= 0) {
    return node->input(index);
  }
  return nullptr;
}

std::shared_ptr<torch::jit::Graph>& FuseCollectiveViewPass::getOriginalGraph() {
  RestoreJITStack(habana_launch_op_ptr_->value_to_ivalue_);
  return original_graph_;
}

void FuseCollectiveViewPass::RelocateJITStack(
    CValuePtrToIValuePtrMap& value_to_ivalue,
    std::shared_ptr<torch::jit::Graph>& graph) {
  std::unordered_map<size_t, IValPtrShared> unique_to_ivalue;

  std::for_each(value_to_ivalue.begin(), value_to_ivalue.end(), [&](auto it) {
    unique_to_ivalue[it.first->unique()] = it.second;
  });

  auto remap_func = [&](const at::ArrayRef<Value*>& values) {
    for (auto value : values) {
      auto it = unique_to_ivalue.find(value->unique());
      if (it != unique_to_ivalue.end()) {
        if (value_to_ivalue.find(value) == value_to_ivalue.end()) {
          value_to_ivalue[value] = it->second;
        }
      }
    }
  };

  for (auto* node : graph->nodes()) {
    remap_func(node->inputs());
  }

  remap_func(graph->inputs());
  remap_func(graph->outputs());
}

void FuseCollectiveViewPass::PrepareJITStack(
    CValuePtrToIValuePtrMap& value_to_ivalue) {
  RelocateJITStack(value_to_ivalue, cloned_graph_);
}

void FuseCollectiveViewPass::RestoreJITStack(
    CValuePtrToIValuePtrMap& value_to_ivalue) {
  RelocateJITStack(value_to_ivalue, original_graph_);
  auto remap_func = [&](const at::ArrayRef<Value*>& src_values,
                        const at::ArrayRef<Value*>& dst_values) {
    for (size_t i = 0; i < src_values.size(); i++) {
      auto it = value_to_ivalue.find(src_values[i]);
      if (it != value_to_ivalue.end()) {
        value_to_ivalue[dst_values[i]] = it->second;
      }
    }
  };

  remap_func(cloned_graph_->inputs(), original_graph_->inputs());
  remap_func(cloned_graph_->outputs(), original_graph_->outputs());
}

bool FuseCollectiveViewPass::CanFuse(CValPtr value, int64_t dim, int64_t step) {
  if (step == 1 && dim == 0) {
    if (auto tensor_type = value->type()->cast<TensorType>()) {
      auto sizes = tensor_type->sizes();
      auto ndim = sizes.size();
      if (ndim.has_value() && ndim.value() >= 1) {
        for (size_t dim = 0; dim < ndim.value(); dim++) {
          if (!sizes[dim].has_value()) {
            return false;
          }
        }
        return true;
      }
    }
  }
  return false;
}

bool FuseCollectiveViewPass::IsGraphInputOutput(
    torch::jit::Value* value,
    bool is_node_output) {
  bool ret = false;

  if (value) {
    auto* graph = value->owningGraph();
    if (graph != nullptr) {
      auto values = is_node_output ? graph->outputs() : graph->inputs();
      auto it = std::find_if(values.begin(), values.end(), [&](Value* value_) {
        return value_ != nullptr && value_->unique() == value->unique();
      });
      if (values.end() != it) {
        ret = true;
      }
    }
  }

  return ret;
}

bool FuseCollectiveViewPass::CanFuse(
    torch::jit::Node* node,
    bool is_node_output) {
  if (node == nullptr) {
    return false;
  }
  if (is_node_output) {
    if (strcmp(node->kind().toQualString(), "hpu::slice_insert") == 0) {
      return IsGraphInputOutput(node->output(), true);
    } else if (strcmp(node->kind().toQualString(), "prim::Return") == 0) {
      return true;
    }
  } else {
    if (strcmp(node->kind().toQualString(), "aten::slice") == 0) {
      auto input = node->input(0);

      auto dim = torch::jit::toIValue(node->input(1));
      auto step = torch::jit::toIValue(node->input(4));

      if (dim.has_value() && step.has_value()) {
        bool can_fuse =
            CanFuse(input, dim.value().toInt(), step.value().toInt());
        if (can_fuse && !IsGraphInputOutput(input)) {
          can_fuse = CanFuse(input->node());
        }
        return can_fuse;
      } else {
        return false;
      }
    } else if (
        strcmp(node->kind().toQualString(), "aten::view") == 0 ||
        strcmp(node->kind().toQualString(), "aten::squeeze") == 0) {
      bool can_fuse = false;
      auto input = node->input(0);
      auto output = node->output();
      auto tensor_type = output->type()->cast<TensorType>();
      auto sizes = tensor_type->sizes();
      auto ndim = sizes.size();
      if (ndim.has_value() && ndim.value() >= 1) {
        auto uses = output->uses();
        auto* successor_node = uses.at(0).user;
        if (successor_node != nullptr &&
            habana_helpers::IsCollective(successor_node->kind())) {
          if (sizes[0].has_value() && ndim.value() == 1) {
            can_fuse = true;
          }
        } else {
          can_fuse = true;
        }
      }
      if (can_fuse && !IsGraphInputOutput(input)) {
        can_fuse = CanFuse(input->node());
      }
      return can_fuse;
    }
  }
  return false;
}

void FuseCollectiveViewPass::GetExternalParams(
    CValPtr value,
    int64_t dim,
    int64_t start,
    int64_t end,
    ExternalParams& params) {
  if (auto tensor_type = value->type()->cast<TensorType>()) {
    auto sizes = tensor_type->sizes();
    auto strides = tensor_type->strides();
    auto ndim = sizes.size();
    auto dim_sizes = sizes[dim];
    auto dim_strides = strides[dim];
    if (ndim.has_value() &&
        ndim.value() > uint64_t(at::maybe_wrap_dim(dim, ndim.value())) &&
        dim_sizes.has_value() && dim_strides.has_value()) {
      auto normalize_func = [](int64_t idx, int64_t size) -> int64_t {
        if (size <= 0) {
          return 0;
        }
        if (idx < -size) {
          idx = 0;
        }
        idx = std::min(idx, size);
        if (idx < 0) {
          idx += size;
        }
        return idx;
      };

      start = normalize_func(start, dim_sizes.value());
      end = normalize_func(end, dim_sizes.value());
      auto stride = dim_strides.value();
      params.offset = start * stride;
      params.numel = (end - start) * stride;
    }
  }
}

void FuseCollectiveViewPass::FuseSliceInsertOps(
    torch::jit::Node* collective_node,
    torch::jit::Value* output,
    torch::jit::Node* slice_insert_node,
    std::vector<torch::jit::Node*>& const_node_vec) {
  auto input = GetInputValue(collective_node, true);
  HABANA_ASSERT(input != nullptr, "Input value can not be null");
  torch::jit::Stack inputs =
      habana_launch_op_ptr_->getStackForNode(slice_insert_node);
  auto it = std::find_if(
      const_node_vec.begin(),
      const_node_vec.end(),
      [&](torch::jit::Node* node) {
        return node->output()->unique() ==
            slice_insert_node->input(2)->unique();
      });
  bool have_shape_tensor = const_node_vec.empty() || const_node_vec.end() == it;
  auto opt_param_list = torch::jit::toIValue(slice_insert_node->input(2));
  if (!have_shape_tensor && opt_param_list.has_value()) {
    auto paramsList = opt_param_list.value().toIntList();
    if (paramsList.size() == 4) {
      ExternalParams params;

      auto dim = paramsList[0];
      auto start = paramsList[1];
      auto end = paramsList[2];
      auto step = paramsList[3];

      bool can_fuse = CanFuse(collective_node->input(0), dim, step);
      if (can_fuse) {
        auto real_output = slice_insert_node->input(0);
        GetExternalParams(real_output, dim, start, end, params);
        input->replaceAllUsesWith(real_output);
        output->replaceAllUsesWith(input);

        habana_launch_op_ptr_->CreateOutputReuseInputSynapseTensor(input);

        output->setType(real_output->type());

        output_valptr_to_params_map_[output] =
            std::make_shared<ExternalParams>(params);

        auto* slice_output = slice_insert_node->output(0);
        slice_output->replaceAllUsesWith(output);
        slice_insert_node->removeAllInputs();
        slice_insert_node->destroy();
      }
    }
  }
}

void FuseCollectiveViewPass::FuseSliceOps(torch::jit::Node* slice_node) {
  auto dim = torch::jit::toIValue(slice_node->input(1));
  auto start = torch::jit::toIValue(slice_node->input(2));
  auto end = torch::jit::toIValue(slice_node->input(3));
  auto step = torch::jit::toIValue(slice_node->input(4));

  if (dim.has_value() && start.has_value() && end.has_value() &&
      step.has_value()) {
    auto input = slice_node->input(0);
    auto output = slice_node->output();
    bool can_fuse = CanFuse(input, dim.value().toInt(), step.value().toInt());

    if (can_fuse) {
      ExternalParams params;
      GetExternalParams(
          input,
          dim.value().toInt(),
          start.value().toInt(),
          end.value().toInt(),
          params);

      input_valptr_to_params_map_[input] =
          std::make_shared<ExternalParams>(params);
      output->replaceAllUsesWith(input);
      slice_node->removeAllInputs();
      slice_node->destroy();
    }
  }
}

void FuseCollectiveViewPass::FuseSqueezeViewOps(torch::jit::Node* node) {
  auto input = node->input(0);
  auto output = node->output(0);

  auto tensor_type = output->type()->cast<TensorType>();
  auto sizes = tensor_type->sizes();
  auto ndim = sizes.size();

  if (ndim.has_value() && ndim.value() >= 1) {
    ExternalParams params;

    auto it = input_valptr_to_params_map_.find(output);
    if (it == input_valptr_to_params_map_.end()) {
      auto first_elem = sizes[0];
      if (first_elem.has_value() && ndim.value() == 1) {
        GetExternalParams(input, 0, 0, first_elem.value(), params);
        input_valptr_to_params_map_[input] =
            std::make_shared<ExternalParams>(params);
        output->replaceAllUsesWith(input);
        node->removeAllInputs();
        node->destroy();
      }
    } else {
      input_valptr_to_params_map_[input] =
          std::make_shared<ExternalParams>(*it->second);
      output->replaceAllUsesWith(input);
      node->removeAllInputs();
      node->destroy();
    }
  }
}

void FuseCollectiveViewPass::RunFuseOps(
    torch::jit::Node* collective_node,
    int index) {
  auto* input = collective_node->input(index);
  if (input != nullptr) {
    auto* node = input->node();
    if (node != nullptr) {
      if (strcmp(node->kind().toQualString(), "aten::slice") == 0) {
        FuseSliceOps(node);
        RunFuseOps(collective_node, index);
      } else if (
          strcmp(node->kind().toQualString(), "aten::view") == 0 ||
          strcmp(node->kind().toQualString(), "aten::squeeze") == 0) {
        FuseSqueezeViewOps(node);
        RunFuseOps(collective_node, index);
      }
    }
  }
}

bool FuseCollectiveViewPass::RunFuseOps(
    torch::jit::graph_node_list graph_nodes,
    bool is_check_mode) {
  std::vector<torch::jit::Node*> collective_node_vec;
  std::vector<torch::jit::Node*> const_node_vec;

  for (auto* node : graph_nodes) {
    if (habana_helpers::IsCollective(node->kind()) &&
        std::get<0>(GetInputOutputIndices(node)) != -1) {
      collective_node_vec.emplace_back(node);
    }

    if (node->kind() == torch::jit::prim::Constant) {
      const_node_vec.emplace_back(node);
    }
  }

  for (auto* collective_node : collective_node_vec) {
    auto indices = GetInputOutputIndices(collective_node);
    auto input_index = std::get<0>(indices);
    if (input_index != -1) {
      auto* input = collective_node->input(input_index);
      if (CanFuse(input->node())) {
        if (is_check_mode) {
          return true;
        } else {
          RunFuseOps(collective_node, input_index);
        }
      }
    }

    for (auto* output : collective_node->outputs()) {
      auto uses = output->uses();
      auto* successor_node = uses.at(0).user;
      if (uses.size() == 2 && CanFuse(successor_node, true) &&
          CanFuse(uses.at(1).user, true)) {
        if (is_check_mode) {
          return true;
        } else {
          FuseSliceInsertOps(
              collective_node, output, successor_node, const_node_vec);
        }
      }
    }
  }

  return false;
}

void FuseCollectiveViewPass::RunFuseOpsPasses(
    const std::shared_ptr<torch::jit::Graph> graph) {
  torch::jit::graph_node_list graph_nodes = graph->nodes();
  RunFuseOps(graph_nodes);
}

void FuseCollectiveViewPass::PatchPTTensorInfo(
    CValPtr value,
    size_t item_size,
    PtTensorInfoShared& ti,
    std::shared_ptr<ExternalParams> params_ptr,
    bool is_input) {
  if (auto tensor_type = value->type()->cast<TensorType>()) {
    auto numel = tensor_type->numel();
    auto external_offset = params_ptr->offset;
    auto external_numel = params_ptr->numel;

    if (numel.has_value() && numel.value() >= external_numel &&
        numel.value() >= external_offset) {
      if (is_input) {
        ti->set_external_numel(external_numel);
      }
      ti->set_external(true);
      ti->set_external_offset(external_offset * item_size);
    }
  }
}

void FuseCollectiveViewPass::ProcessInputPTTensorInfo(
    std::unordered_map<CValPtr, std::shared_ptr<ExternalParams>>&
        valptr_to_params_map,
    torch::jit::Node* node,
    habana_helpers::CollectiveKernelInfos::Info& kernel_info) {
  auto node_inputs = node->inputs();
  for (size_t i = 0; i < node_inputs.size(); i++) {
    auto input = node_inputs[i];
    if (valptr_to_params_map.find(input) != valptr_to_params_map.end()) {
      auto params_ptr = valptr_to_params_map[input];
      if (params_ptr != nullptr) {
        auto indices = GetInputOutputIndices(node);
        int in_val_idx = std::get<0>(indices);
        PtTensorInfoShared& ti = kernel_info.input_tensor_infos[in_val_idx];
        habana::HabanaOperatorPtr kernel = kernel_info.kernel;
        if (ti != nullptr) {
          auto& in_tensor = kernel->GetInputs()[i];
          HABANA_ASSERT(
              !habana::is_ZST(in_tensor), "Input tensor can not be ZST");
          PatchPTTensorInfo(input, in_tensor.itemsize(), ti, params_ptr);
        }
      }
    }
  }
}

void FuseCollectiveViewPass::ProcessOutputPTTensorInfo(
    std::unordered_map<CValPtr, std::shared_ptr<ExternalParams>>&
        valptr_to_params_map,
    torch::jit::Node* node,
    habana_helpers::CollectiveKernelInfos::Info& kernel_info) {
  auto node_outputs = node->outputs();
  for (size_t i = 0; i < node_outputs.size(); i++) {
    auto output = node_outputs[i];
    if (valptr_to_params_map.find(output) != valptr_to_params_map.end()) {
      auto params_ptr = valptr_to_params_map[output];
      if (params_ptr != nullptr) {
        auto indices = GetInputOutputIndices(node);
        int in_val_idx = std::get<1>(indices);
        PtTensorInfoShared& ti = kernel_info.input_tensor_infos[in_val_idx];
        habana::HabanaOperatorPtr kernel = kernel_info.kernel;
        if (ti != nullptr) {
          auto& out_tensor = kernel->GetOutputs()[i];
          HABANA_ASSERT(
              !habana::is_ZST(out_tensor), "Output tensor can not be ZST");
          PatchPTTensorInfo(
              output, out_tensor.itemsize(), ti, params_ptr, false);
        }
      }
    }
  }
}

void FuseCollectiveViewPass::PostRunFuseOpsPasses(
    torch::jit::Node* node,
    habana_helpers::CollectiveKernelInfos::Info& kernel_info) {
  ProcessInputPTTensorInfo(getInputValPtrToParamsMap(), node, kernel_info);
  ProcessOutputPTTensorInfo(getOutputValPtrToParamsMap(), node, kernel_info);
}

bool FuseCollectiveViewPass::NeedCheck(
    std::shared_ptr<torch::jit::Graph> graph) {
  torch::jit::graph_node_list graph_nodes = graph->nodes();
  return RunFuseOps(graph_nodes, true);
}

std::shared_ptr<torch::jit::Graph> FuseCollectiveViewPass::CreateClonedGraph(
    std::shared_ptr<torch::jit::Graph> graph) {
  if (NeedCheck(graph)) {
    original_graph_ = graph;
    cloned_graph_ = graph->copy();
    PrepareJITStack(habana_launch_op_ptr_->value_to_ivalue_);
    return cloned_graph_;
  }

  return nullptr;
}

std::unique_ptr<FuseCollectiveViewPassData> FuseCollectiveViewPass::VisitGraph(
    const std::shared_ptr<torch::jit::Graph> graph) {
  HABANA_ASSERT(nullptr != habana_launch_op_ptr_);
  HABANA_ASSERT(nullptr != graph.get());
  auto cloned_graph_ptr_sh = CreateClonedGraph(graph);
  if (cloned_graph_ptr_sh != nullptr) {
    RunFuseOpsPasses(cloned_graph_ptr_sh);
    return std::make_unique<FuseCollectiveViewPassData>(
        std::make_shared<FuseCollectiveViewPass>(*this));
  }

  return nullptr;
}
