/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include <torch/csrc/jit/ir/ir.h>
#include <memory>
#include <string>
#include <vector>
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "habana_eager/graph_dynamic.h"
#include "habana_eager/graph_dynamic_ops.h"

namespace habana {
namespace graph {

bool ViewOperatorDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* aten_view_node,
    torch::jit::Stack& org_stack,
    GraphInputIndexMap& org_stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  HABANA_ASSERT(2 == aten_view_node->inputs().size());
  static const auto list_construct_symbol{
      c10::Symbol::fromQualString("prim::ListConstruct")};
  auto graph{aten_view_node->owningGraph()};
  auto v_view_shape = aten_view_node->inputs().at(1);
  HABANA_ASSERT(
      v_view_shape->node()->kind() == list_construct_symbol,
      "View input is not a ListConstruct, it is: ",
      v_view_shape->node()->kind().toQualString());
  auto list_construct_node{v_view_shape->node()};

  // Step 1: Collect shape and scalar pos used in ListConstruct input node
  std::vector<int64_t> st_size;
  std::vector<int64_t> scalar_indexes;
  GetValuesAndScalarIndexesFromListConstruct(
      list_construct_node,
      org_stack,
      org_stack_index_map,
      st_size,
      scalar_indexes);

  // Use actual reshape sizes and avoid sizes with dims "-1"
  auto out_tensors = getOutputTensers(aten_view_node, value_ivalue_map);
  auto inferred_st_sizes = out_tensors[0].sizes().vec();
  // Step2: Create shape tensor and insert to graph inputs.
  auto view_st_name =
      GetDynamicTensorName(v_view_shape->debugName(), SHAPE_TENSOR);
  int64_t stack_index =
      CreateSTAndInsertToDSStack(inferred_st_sizes, scalar_indexes, m_dmeta);
  auto v_st_tensor = graph->addInput(view_st_name);

  // find if view has negative dims
  auto has_neg_size = false;
  for (auto val : st_size) {
    if (val < 0) {
      has_neg_size = true;
      break;
    }
  }

  // Step3: Create hpu::view node and insert to the graph
  torch::jit::Node* hpu_view_node;
  if (has_neg_size) {
    static const auto hpu_view_symbol{
        c10::Symbol::fromQualString("hpu::view_neg")};
    hpu_view_node = CreateAndInsertDynamicNodeToGraph(
        graph,
        aten_view_node,
        hpu_view_symbol,
        {aten_view_node->input(0), v_st_tensor, v_view_shape},
        value_ivalue_map);
  } else {
    static const auto hpu_view_symbol{c10::Symbol::fromQualString("hpu::view")};
    hpu_view_node = CreateAndInsertDynamicNodeToGraph(
        graph,
        aten_view_node,
        hpu_view_symbol,
        {aten_view_node->input(0), v_st_tensor},
        value_ivalue_map);
  }

  // Step4: Register patching function and tensor lists
  std::vector<int64_t> dtensor_indexes{stack_index};
  InputPatchPair patch_info(&DynamicOp::UpdateDynamicInputs, dtensor_indexes);
  m_dmeta->ds_input_patching_list.push_back(patch_info);
  if (has_neg_size)
    m_dmeta->negative_size_nodes.emplace_back(hpu_view_node);

  return true;
}

void ViewOperatorDS::ResolveNegativeSizes(
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& org_stack,
    torch::jit::Node* node,
    std::unordered_map<CValPtr, torch::jit::IValue>& value_ivalue_map) {
  ValueIvalueMap gin_value_ivalue_map;
  for (size_t j = 0; j < org_stack.size(); j++) {
    auto value_input = graph->inputs().at(j);
    auto ivpsh = std::make_shared<IVal>(org_stack[j]);
    gin_value_ivalue_map[value_input] = ivpsh;
  }
  auto view_st_value = node->inputs().at(1);
  auto view_out_value = node->outputs().at(0);
  auto cos_t_shapes = value_ivalue_map[view_out_value].toTensor().sizes().vec();
  auto ivsh_view_st = gin_value_ivalue_map[view_st_value];
  ivsh_view_st->toTensor().unsafeGetTensorImpl()->set_sizes_contiguous(
      cos_t_shapes);
}

bool IsStridedRatioUndefined(
    std::vector<int64_t>& self_strides,
    std::vector<int64_t>& stride_sizes) {
  if (self_strides.size() != stride_sizes.size()) {
    return true;
  }
  auto len = self_strides.size();
  for (uint64_t i = 0; i < len; i++) {
    if (stride_sizes[i] < self_strides[i]) {
      return true;
    }
  }
  return false;
}

bool AsStridedOperatorDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* aten_as_strided_node,
    torch::jit::Stack& in_stack,
    GraphInputIndexMap& org_stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  HABANA_ASSERT(4 == aten_as_strided_node->inputs().size());
  auto in_tensors = getInputTensers(aten_as_strided_node, value_ivalue_map);
  auto self = in_tensors[0];
  auto as_strided_shape = aten_as_strided_node->inputs().at(1);
  auto as_strided_stride = aten_as_strided_node->inputs().at(2);
  auto as_strided_offset = aten_as_strided_node->inputs().at(3);
  auto graph{aten_as_strided_node->owningGraph()};
  auto shape_construct_node{as_strided_shape->node()};
  auto stride_construct_node{as_strided_stride->node()};
  auto offset_construct_node{as_strided_offset->node()};
  static const auto hpu_as_strided_orig_symbol{
      c10::Symbol::fromQualString("hpu::strided_view_orig_ds_h2d")};
  static const auto hpu_as_strided_symbol{
      c10::Symbol::fromQualString("hpu::strided_view_ds_h2d")};

  // Collect ST shape and symlnt pos using ListConstruct values for sizes
  std::vector<int64_t> values_shapes;
  std::vector<int64_t> scalar_indexes_shape;
  if (shape_construct_node->kind() == torch::jit::prim::Constant) {
    GetValuesAndScalarIndexesFromListConst(
        shape_construct_node, values_shapes, scalar_indexes_shape);
  } else if (shape_construct_node->kind() == torch::jit::prim::ListConstruct) {
    GetValuesAndScalarIndexesFromListConstruct(
        shape_construct_node,
        in_stack,
        org_stack_index_map,
        values_shapes,
        scalar_indexes_shape);
  }
  auto as_strided_shape_st_name =
      GetDynamicTensorName(as_strided_shape->debugName(), SHAPE_TENSOR);
  int64_t stack_index =
      CreateSTAndInsertToDSStack(values_shapes, scalar_indexes_shape, m_dmeta);
  auto v_st_sizes_tensor = graph->addInput(as_strided_shape_st_name);
  std::vector<int64_t> dtensor_indexes{stack_index};

  // Collect ST shape and symlnt pos using ListConstruct values for strides
  // format of filling [num_strides, offset, stride[0], stride[1]...]
  auto as_strided_stride_st_name = GetDynamicTensorName(
      as_strided_stride->debugName(), HOST_TO_DEVICE_TENSOR);
  std::vector<int64_t> scalar_indexes;
  std::vector<uint64_t> h2d_values;
  // Get offset value
  int64_t offset_idx = LONG_MAX;
  int64_t offset_value = 0;
  GetValueAndScalarIndexFromInput(
      as_strided_offset,
      in_stack,
      org_stack_index_map,
      offset_value,
      offset_idx);
  // Fill the offset values
  scalar_indexes.push_back(offset_idx);
  h2d_values.push_back(static_cast<uint64_t>(offset_value));
  // Get and fill strides values
  std::vector<int64_t> scalar_indexes_strides;
  std::vector<int64_t> values_strides;
  auto self_strides = self.strides().vec();
  if (stride_construct_node->kind() == torch::jit::prim::Constant) {
    GetValuesAndScalarIndexesFromListConst(
        stride_construct_node, values_strides, scalar_indexes_strides);
  } else if (stride_construct_node->kind() == torch::jit::prim::ListConstruct) {
    GetValuesAndScalarIndexesFromListConstruct(
        stride_construct_node,
        in_stack,
        org_stack_index_map,
        values_strides,
        scalar_indexes_strides);
  }
  // Fill the strides values in reverse order
  for (auto it = values_strides.rbegin(); it != values_strides.rend(); ++it) {
    h2d_values.push_back(static_cast<uint64_t>(*it));
  }
  // Since strides are reversed fill strides indexes also in reverse order
  for (auto it = scalar_indexes_strides.rbegin();
       it != scalar_indexes_strides.rend();
       ++it) {
    scalar_indexes.push_back(*it);
  }
  // Fill the remaining with 0
  size_t fill_dim = (SYN_MAX_TENSOR_DIM + 1) - values_strides.size();
  for (size_t i = 0; i < fill_dim; i++) {
    h2d_values.push_back(static_cast<uint64_t>(0));
    scalar_indexes.push_back(LONG_MAX);
  }
  // Fill num_strides at 0 index
  scalar_indexes.insert(scalar_indexes.begin(), LONG_MAX);
  h2d_values.insert(
      h2d_values.begin(), static_cast<uint64_t>(values_strides.size()));

  at::Tensor h2d_tensor_strides = createDynamicTensor(
      {static_cast<int64_t>(h2d_values.size()) * 2}, HOST_TO_DEVICE_TENSOR);
  SetH2DTensorHostData<uint64_t>(
      h2d_tensor_strides, h2d_values, HostDataType::UINT64_T, false);
  auto iv_st_strides_tensor = torch::jit::IValue(h2d_tensor_strides);
  int64_t stack_index_strides =
      UpdateDynamicTensorDSStack(iv_st_strides_tensor, scalar_indexes, m_dmeta);
  auto v_st_strides_tensor = graph->addInput(as_strided_stride_st_name);
  dtensor_indexes.push_back(stack_index_strides);

  if (IsStridedRatioUndefined(self_strides, values_strides)) {
    auto tmeta{get_tensor_extra_meta(h2d_tensor_strides)};
    tmeta->set_H2D_data_for_bucketing();
    // Create hpu::as_strided_view node and insert to the graph
    CreateAndInsertDynamicNodeToGraph(
        graph,
        aten_as_strided_node,
        hpu_as_strided_orig_symbol,
        {aten_as_strided_node->input(0),
         v_st_sizes_tensor,
         v_st_strides_tensor},
        value_ivalue_map);
  } else {
    std::vector<int64_t> values_offset;
    values_offset.push_back(offset_value);
    at::Tensor st_tensor_offset =
        createDynamicTensor(values_offset, SHAPE_TENSOR);
    auto tmeta_offset{get_tensor_extra_meta(st_tensor_offset)};
    // Mark this front end shape tensor as it does not need synapse tensor.
    // It carries stride_ratios info for BE lowering kernel.
    tmeta_offset->set_H2D_frontend_shape_tensor();
    std::vector<int64_t> stride_ratios;
    auto stride_sizes = values_strides;
    auto len = stride_sizes.size();
    for (uint64_t i = 0; i < len; i++) {
      stride_ratios.push_back(stride_sizes[i] / self_strides[i]);
    }
    tmeta_offset->get_shape_struct().set_strides_tensor_shape(stride_sizes);
    tmeta_offset->get_shape_struct().set_stride_ratio(stride_ratios);
    PT_DYNAMIC_SHAPE_DEBUG(
        "Frontend self strides = ",
        self_strides,
        " view strides = ",
        stride_sizes);
    PT_DYNAMIC_SHAPE_DEBUG(
        "Setting stride ratio = ", stride_ratios, " offset = ", offset_value);
    auto iv_st_offset_tensor = torch::jit::IValue(st_tensor_offset);
    std::vector<int64_t> scalar_indexes_offset;
    scalar_indexes_offset.push_back(offset_idx);
    int64_t stack_index_offset = UpdateDynamicTensorDSStack(
        iv_st_offset_tensor, scalar_indexes_offset, m_dmeta);
    auto as_strided_offset_st_name =
        GetDynamicTensorName(as_strided_offset->debugName(), SHAPE_TENSOR);
    auto v_st_offset_tensor = graph->addInput(as_strided_offset_st_name);
    dtensor_indexes.push_back(stack_index_offset);
    // Create hpu::as_strided_view node and insert to the graph
    CreateAndInsertDynamicNodeToGraph(
        graph,
        aten_as_strided_node,
        hpu_as_strided_symbol,
        {aten_as_strided_node->input(0),
         v_st_sizes_tensor,
         v_st_strides_tensor,
         v_st_offset_tensor},
        value_ivalue_map);
  }
  InputPatchPair patch_info(
      &AsStridedOperatorDS::UpdateDynamicInputs, dtensor_indexes);
  m_dmeta->ds_input_patching_list.push_back(patch_info);
  return true;
}

void AsStridedOperatorDS::UpdateDynamicInputs(
    c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
    c10::SmallVectorImpl<habana::graph::SymIntData>& scalar_idx_list,
    std::vector<c10::IValue>& orig_stack) {
  HABANA_ASSERT(
      dtensor_list.size() == scalar_idx_list.size(),
      "Dtensor and SymIntData count not matching");
  HABANA_ASSERT(
      dtensor_list.size() == scalar_idx_list.size(),
      "Dtensor and SymIntData count not matching");

  // Stride H2D patching
  // Format of H2D [num_strides, offset, stride[n], stride[n-1], .., 0]
  auto dtensor = dtensor_list[1]->toTensor();
  SymIntData& scalar_idx = scalar_idx_list[1];

  std::vector<uint64_t> updated_h2d_data;
  for (int idx = 0; idx < scalar_idx.values.size(); idx++) {
    auto stack_index = scalar_idx.values[idx];
    if (stack_index == LONG_MAX) {
      std::vector<uint64_t> h2d_data = GetH2DTensorHostData<uint64_t>(dtensor);
      updated_h2d_data.push_back(h2d_data[idx]);
    } else {
      updated_h2d_data.push_back(
          static_cast<uint64_t>(GetSymintValue(orig_stack, stack_index)));
    }
  }

  UpdateH2DTensorData<uint64_t>(dtensor, updated_h2d_data);

  for (auto i = 0; i < dtensor_list.size(); i++) {
    if (i == 1)
      continue;
    auto dtensor = dtensor_list[i]->toTensor();
    SymIntData st_values = scalar_idx_list[i];
    UpdateShapeTensorSize(dtensor, st_values.values, orig_stack);
  }
  // offset patching in case stridedratio flow is used
  // Need to update the stride information also to be patched
  // in case of dynamic cache hit
  // Read the relevant data from updated_h2d_data and update the tmeta
  // @TODO - Check if need to recalculate the stride ratio also
  if (dtensor_list.size() == 3) {
    auto dtensor_offset = dtensor_list[2]->toTensor();
    auto tmeta_offset{get_tensor_extra_meta(dtensor_offset)};
    auto num_strides = updated_h2d_data[0];
    std::vector<int64_t> actual_strides;
    for (auto i = 2; i < (2 + num_strides); i++) {
      actual_strides.push_back(updated_h2d_data[i]);
    }
    // Since strides here are reversed, make it unreverse
    std::reverse(actual_strides.begin(), actual_strides.end());
    tmeta_offset->get_shape_struct().set_strides_tensor_shape(actual_strides);
  }
}

bool StridedInsertOperatorDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* strided_insert_node,
    torch::jit::Stack& in_stack,
    GraphInputIndexMap& org_stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  HABANA_ASSERT(4 == strided_insert_node->inputs().size());
  auto in_tensors = getInputTensers(strided_insert_node, value_ivalue_map);
  auto self = in_tensors[0];
  auto strided_insert_stride = strided_insert_node->inputs().at(2);
  auto strided_insert_offset = strided_insert_node->inputs().at(3);
  static const auto hpu_strided_insert_orig_symbol{
      c10::Symbol::fromQualString("hpu::strided_insert_orig_ds_h2d")};
  static const auto hpu_strided_insert_symbol{
      c10::Symbol::fromQualString("hpu::strided_insert_orig_ds")};
  auto stride_construct_node{strided_insert_stride->node()};
  auto offset_construct_node{strided_insert_offset->node()};
  auto graph{strided_insert_node->owningGraph()};
  std::vector<int64_t> dtensor_indexes;
  // Collect ST shape and symlnt pos using ListConstruct values for strides
  // format of filling [num_strides, offset, stride[0], stride[1]...]
  auto strided_insert_stride_st_name = GetDynamicTensorName(
      strided_insert_stride->debugName(), HOST_TO_DEVICE_TENSOR);
  std::vector<int64_t> scalar_indexes;
  std::vector<uint64_t> h2d_values;
  // Get offset value
  int64_t offset_idx = LONG_MAX;
  int64_t offset_value = 0;
  GetValueAndScalarIndexFromInput(
      strided_insert_offset,
      in_stack,
      org_stack_index_map,
      offset_value,
      offset_idx);
  // Fill the offset values
  scalar_indexes.push_back(offset_idx);
  h2d_values.push_back(static_cast<uint64_t>(offset_value));
  // Get and fill strides values
  std::vector<int64_t> scalar_indexes_strides;
  std::vector<int64_t> values_strides;
  auto self_strides = self.strides().vec();
  GetValuesAndScalarIndexesFromListConstruct(
      stride_construct_node,
      in_stack,
      org_stack_index_map,
      values_strides,
      scalar_indexes_strides);
  // Fill the strides values in reverse order
  for (auto it = values_strides.rbegin(); it != values_strides.rend(); ++it) {
    h2d_values.push_back(static_cast<uint64_t>(*it));
  }
  // Since strides are reversed fill strides indexes also in reverse order
  for (auto it = scalar_indexes_strides.rbegin();
       it != scalar_indexes_strides.rend();
       ++it) {
    scalar_indexes.push_back(*it);
  }
  // Fill the remaining with 0
  size_t fill_dim = (SYN_MAX_TENSOR_DIM + 1) - values_strides.size();
  for (size_t i = 0; i < fill_dim; i++) {
    h2d_values.push_back(static_cast<uint64_t>(0));
    scalar_indexes.push_back(LONG_MAX);
  }
  // Fill num_strides at 0 index
  scalar_indexes.insert(scalar_indexes.begin(), LONG_MAX);
  h2d_values.insert(
      h2d_values.begin(), static_cast<uint64_t>(values_strides.size()));

  at::Tensor h2d_tensor_strides = createDynamicTensor(
      {static_cast<int64_t>(h2d_values.size()) * 2}, HOST_TO_DEVICE_TENSOR);
  SetH2DTensorHostData<uint64_t>(
      h2d_tensor_strides, h2d_values, HostDataType::UINT64_T, false);
  auto iv_st_strides_tensor = torch::jit::IValue(h2d_tensor_strides);
  int64_t stack_index_strides =
      UpdateDynamicTensorDSStack(iv_st_strides_tensor, scalar_indexes, m_dmeta);
  auto v_st_strides_tensor = graph->addInput(strided_insert_stride_st_name);
  dtensor_indexes.push_back(stack_index_strides);

  if (IsStridedRatioUndefined(self_strides, values_strides)) {
    auto tmeta{get_tensor_extra_meta(h2d_tensor_strides)};
    tmeta->set_H2D_data_for_bucketing();
    // Create hpu::as_strided_view node and insert to the graph
    CreateAndInsertDynamicNodeToGraph(
        graph,
        strided_insert_node,
        hpu_strided_insert_orig_symbol,
        {strided_insert_node->input(0),
         strided_insert_node->input(1),
         v_st_strides_tensor},
        value_ivalue_map);
  } else {
    std::vector<int64_t> values_offset;
    values_offset.push_back(offset_value);
    at::Tensor st_tensor_offset =
        createDynamicTensor(values_offset, SHAPE_TENSOR);
    auto tmeta_offset{get_tensor_extra_meta(st_tensor_offset)};
    // Mark this front end shape tensor as it does not need synapse tensor.
    // It carries stride_ratios info for BE lowering kernel.
    tmeta_offset->set_H2D_frontend_shape_tensor();
    std::vector<int64_t> stride_ratios;
    auto stride_sizes = values_strides;
    auto len = stride_sizes.size();
    for (uint64_t i = 0; i < len; i++) {
      stride_ratios.push_back(stride_sizes[i] / self_strides[i]);
    }
    tmeta_offset->get_shape_struct().set_strides_tensor_shape(stride_sizes);
    tmeta_offset->get_shape_struct().set_stride_ratio(stride_ratios);
    PT_DYNAMIC_SHAPE_DEBUG(
        "Setting stride ratio = ", stride_ratios, " offset = ", offset_value);
    auto iv_st_offset_tensor = torch::jit::IValue(st_tensor_offset);
    std::vector<int64_t> scalar_indexes_offset;
    scalar_indexes_offset.push_back(offset_idx);
    int64_t stack_index_offset = UpdateDynamicTensorDSStack(
        iv_st_offset_tensor, scalar_indexes_offset, m_dmeta);
    auto strided_insert_offset_st_name =
        GetDynamicTensorName(strided_insert_offset->debugName(), SHAPE_TENSOR);
    auto v_st_offset_tensor = graph->addInput(strided_insert_offset_st_name);
    dtensor_indexes.push_back(stack_index_offset);
    // Create hpu::as_strided_view node and insert to the graph
    CreateAndInsertDynamicNodeToGraph(
        graph,
        strided_insert_node,
        hpu_strided_insert_symbol,
        {strided_insert_node->input(0),
         strided_insert_node->input(1),
         v_st_strides_tensor,
         v_st_offset_tensor},
        value_ivalue_map);
  }
  InputPatchPair patch_info(
      &StridedInsertOperatorDS::UpdateDynamicInputs, dtensor_indexes);
  m_dmeta->ds_input_patching_list.push_back(patch_info);
  return true;
}

void StridedInsertOperatorDS::UpdateDynamicInputs(
    c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
    c10::SmallVectorImpl<habana::graph::SymIntData>& scalar_idx_list,
    std::vector<c10::IValue>& orig_stack) {
  HABANA_ASSERT(
      dtensor_list.size() == scalar_idx_list.size(),
      "Dtensor and SymIntData count not matching");
  HABANA_ASSERT(
      dtensor_list.size() == scalar_idx_list.size(),
      "Dtensor and SymIntData count not matching");

  // Stride H2D patching
  // Format of H2D [num_strides, offset, stride[n], stride[n-1], .., 0]
  auto dtensor = dtensor_list[0]->toTensor();
  SymIntData& scalar_idx = scalar_idx_list[0];

  std::vector<uint64_t> updated_h2d_data;
  for (int idx = 0; idx < scalar_idx.values.size(); idx++) {
    auto stack_index = scalar_idx.values[idx];
    if (stack_index == LONG_MAX) {
      std::vector<uint64_t> h2d_data = GetH2DTensorHostData<uint64_t>(dtensor);
      updated_h2d_data.push_back(h2d_data[idx]);
    } else {
      updated_h2d_data.push_back(
          static_cast<uint64_t>(GetSymintValue(orig_stack, stack_index)));
    }
  }
  UpdateH2DTensorData<uint64_t>(dtensor, updated_h2d_data);

  // offset patching in case stridedratio flow is used
  // Need to update the stride information also to be patched
  // in case of dynamic cache hit
  // Read the relevant data from updated_h2d_data and update the tmeta
  // @TODO - Check if need to recalculate the stride ratio also
  if (dtensor_list.size() == 2) {
    auto dtensor = dtensor_list[1]->toTensor();
    SymIntData st_values = scalar_idx_list[1];
    UpdateShapeTensorSize(dtensor, st_values.values, orig_stack);
    auto dtensor_offset = dtensor_list[1]->toTensor();
    auto tmeta_offset{get_tensor_extra_meta(dtensor_offset)};
    auto num_strides = updated_h2d_data[0];
    std::vector<int64_t> actual_strides;
    for (auto i = 2; i < (2 + num_strides); i++) {
      actual_strides.push_back(updated_h2d_data[i]);
    }
    // Since strides here are reversed, make it unreverse
    std::reverse(actual_strides.begin(), actual_strides.end());
    tmeta_offset->get_shape_struct().set_strides_tensor_shape(actual_strides);
  }
}

} // namespace graph
} // namespace habana
