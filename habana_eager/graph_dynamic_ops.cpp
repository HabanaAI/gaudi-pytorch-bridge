/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#include <sstream>
#include "common/utils.h"

#include "habana_eager/graph_dynamic_ops.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "habana_eager/graph_dynamic.h"

#include "habana_helpers/logging.h"
#include "habana_kernels/index_kernels.h"

namespace habana {
namespace graph {

void GetValueAndScalarIndexFromInput(
    torch::jit::Value* input,
    torch::jit::Stack& in_stack,
    GraphInputIndexMap& org_stack_index_map,
    int64_t& value,
    int64_t& index) {
  if (input == nullptr)
    return;
  static const auto constant_symbol{
      c10::Symbol::fromQualString("prim::Constant")};
  static const auto value_attr{torch::jit::Symbol::attr("value")};
  auto in_name = input->debugName();
  if (input->node()->kind() == constant_symbol) {
    try {
      value = static_cast<int64_t>(input->node()->i(value_attr));
      if (value < 0)
        index = value;
    } catch (std::exception& e) {
      value = 0;
    }
  } else if (org_stack_index_map.count(in_name)) {
    index = static_cast<int64_t>(org_stack_index_map[in_name]);
    value = static_cast<int64_t>(in_stack[index].toScalar().toLong());
  } else {
    HABANA_ASSERT(
        false,
        "Node input=",
        in_name,
        " is not a graph input or a prim::Constant");
  }

  PT_EAGER_DEBUG(
      "Value and index of an input:",
      in_name,
      ", is value=",
      value,
      " index=",
      index);
}

void GetValuesAndScalarIndexesFromListConst(
    torch::jit::Node* node,
    std::vector<int64_t>& values,
    std::vector<int64_t>& scalar_indexes) {
  static const auto list_const_symbol{
      c10::Symbol::fromQualString("prim::Constant")};
  HABANA_ASSERT(
      node->kind() == list_const_symbol,
      "input is not a Constant, it is: ",
      node->kind().toQualString(),
      " node: ",
      *node);

  auto value = node->output(0);
  torch::jit::IValue const_ivalue = torch::jit::toIValue(value).value();
  if (const_ivalue.isIntList()) {
    int64_t input_idx = LONG_MAX;
    auto vec = const_ivalue.toIntVector();
    for (auto v : vec) {
      scalar_indexes.push_back(input_idx);
      values.push_back(v);
    }
  } else {
    HABANA_ASSERT(false, "input is not Const Ints..");
  }
}

std::string GetRangeInfoExprFromInput(
    torch::jit::Value* input,
    GraphInputIndexMap& org_stack_index_map,
    std::vector<habana_helpers::RangeInfo>* range_infos) {
  std::string value = "0";
  if (input == nullptr)
    return value;
  static const auto constant_symbol{
      c10::Symbol::fromQualString("prim::Constant")};
  static const auto value_attr{torch::jit::Symbol::attr("value")};
  auto in_name = input->debugName();
  if (input->node()->kind() == constant_symbol) {
    try {
      value =
          std::to_string(static_cast<int64_t>(input->node()->i(value_attr)));
    } catch (std::exception& e) {
      // Sometimes when value in prim::Constant node should have 0 as value_attr
      // set but seems its coming as NoneType =
      // prim::Constant[deterministic=0]() in some case  which we are internally
      // treating as 0
      PT_DYNAMIC_SHAPE_WARN(
          "Node ",
          in_name,
          " does not have value_attr set, treating the value as 0");
      value = "0";
    }
  } else if (org_stack_index_map.count(in_name)) {
    auto index = static_cast<int64_t>(org_stack_index_map[in_name]);
    value = range_infos->at(index).expr;
  } else {
    HABANA_ASSERT(
        false,
        "Node input=",
        in_name,
        " is not a graph input or a prim::Constant");
  }

  PT_DYNAMIC_SHAPE_DEBUG("Input:", in_name, ", expr value=", value);
  return value;
}

std::vector<std::string> GetRangeInfoExprFromListConstruct(
    torch::jit::Node* node,
    GraphInputIndexMap& org_stack_index_map,
    std::vector<habana_helpers::RangeInfo>* range_infos) {
  static const auto list_construct_symbol{
      c10::Symbol::fromQualString("prim::ListConstruct")};
  HABANA_ASSERT(
      node->kind() == list_construct_symbol,
      "input is not a ListConstruct, it is: ",
      node->kind().toQualString());
  std::vector<std::string> expr_values;
  for (auto input : node->inputs()) {
    auto expr_value =
        GetRangeInfoExprFromInput(input, org_stack_index_map, range_infos);
    expr_values.push_back(expr_value);
  }

  return expr_values;
}

std::vector<std::string> GetRangeInfoExprFromListConst(
    torch::jit::Node* node,
    GraphInputIndexMap& org_stack_index_map,
    std::vector<habana_helpers::RangeInfo>* range_infos) {
  static_cast<void>(range_infos);
  static_cast<void>(org_stack_index_map);
  static const auto list_const_symbol{
      c10::Symbol::fromQualString("prim::Constant")};
  HABANA_ASSERT(
      node->kind() == list_const_symbol,
      "input is not a Constant, it is: ",
      node->kind().toQualString(),
      " node: ",
      *node);

  auto value = node->output(0);
  std::vector<std::string> expr_values;
  torch::jit::IValue const_ivalue = torch::jit::toIValue(value).value();
  if (const_ivalue.isIntList()) {
    auto vec = const_ivalue.toIntVector();
    for (auto v : vec) {
      expr_values.push_back(std::to_string(v));
    }
  } else {
    HABANA_ASSERT(false, "input is not Const Ints..");
  }
  return expr_values;
}

std::string GetExprFromString(std::vector<std::string> inputs) {
  std::ostringstream expr;
  expr << '[';
  for (size_t i = 0; i < inputs.size(); i++) {
    expr << (i > 0 ? ", " : "") << inputs[i];
  }
  expr << ']';
  std::string result = expr.str();
  PT_DYNAMIC_SHAPE_DEBUG("Adding Expr string :", result);
  return result;
}

void GetValuesAndScalarIndexesFromListConstruct(
    torch::jit::Node* node,
    torch::jit::Stack& in_stack,
    GraphInputIndexMap& org_stack_index_map,
    std::vector<int64_t>& values,
    std::vector<int64_t>& scalar_indexes) {
  static const auto list_construct_symbol{
      c10::Symbol::fromQualString("prim::ListConstruct")};
  HABANA_ASSERT(
      node->kind() == list_construct_symbol,
      "input is not a ListConstruct, it is: ",
      node->kind().toQualString());

  for (auto input : node->inputs()) {
    int64_t input_idx = LONG_MAX;
    int64_t act_value = 0;
    GetValueAndScalarIndexFromInput(
        input, in_stack, org_stack_index_map, act_value, input_idx);
    scalar_indexes.push_back(input_idx);
    values.push_back(act_value);
  }
}

torch::jit::Node* CreateAndInsertDynamicNodeToGraph(
    torch::jit::Graph* graph,
    torch::jit::Node* aten_node,
    const c10::Symbol& hpu_symbol,
    c10::ArrayRef<torch::jit::Value*> inputs,
    ValueIvalueMap& value_ivalue_map) {
  torch::jit::WithInsertPoint insert_guard{aten_node};
  auto hpu_node{graph->insertNode(graph->create(hpu_symbol, inputs, 0))};
  int output_count = 0;
  for (auto output : aten_node->outputs()) {
    hpu_node->addOutput()->copyMetadata(output);
    output->replaceAllUsesAfterNodeWith(
        hpu_node, hpu_node->output(output_count));
    value_ivalue_map[hpu_node->output(output_count)] = value_ivalue_map[output];
    output_count = output_count + 1;
  }

  if (GET_ENV_FLAG_NEW(PT_HPU_OPTIM_DYNAMIC_OUTPUT_SIF)) {
    auto symbol_outputshape = c10::Symbol::attr("output_shapes");
    if (aten_node->hasAttribute(symbol_outputshape)) {
      hpu_node->s_(symbol_outputshape, aten_node->s(symbol_outputshape));
    } else {
      auto node_qual_str = aten_node->kind().toQualString();
      PT_EAGER_DEBUG("Output Shape is missing for node = ", node_qual_str)
      hpu_node->s_(symbol_outputshape, "[[]]");
    }
  }

  hpu_node->i_(
      torch::jit::attr::deterministic,
      aten_node->i(torch::jit::attr::deterministic));

  return hpu_node;
}

// TODO SW-152611
void UpdateShapeTensorSize(
    at::Tensor& dtensor,
    std::vector<int64_t>& stack_idxs,
    std::vector<c10::IValue>& orig_stack,
    LaunchDynamicShapes& launch_shapes) {
  c10::SmallVector<int64_t, NUM_TENSOR_DIMS> new_shape(stack_idxs.size(), 1);

  for (size_t idx = 0; idx < stack_idxs.size(); ++idx) {
    auto stack_index = stack_idxs[idx];
    if (stack_index == LONG_MAX) {
      new_shape[idx] = dtensor.sizes()[idx];
    } else if (stack_index < 0) {
      // add support for negative consts.. empty the shape tensor and COS in Sif
      new_shape.set_size(0);
      break;
    } else {
      new_shape[idx] = static_cast<int64_t>(GetSymintValue(orig_stack, stack_index));
    }
  }

  PT_EAGER_DEBUG("Updated dynamic shape tensor size:", dtensor.sizes());
  std::vector<int64_t> cast_shapes(new_shape.begin(), new_shape.end());
  launch_shapes.ds_tensors.push_back(dtensor);
  launch_shapes.patch_values.push_back(cast_shapes);
}

void UpdateH2DPatchingData(
    at::Tensor& dtensor,
    std::vector<int64_t>& data,
    LaunchDynamicShapes& launch_shapes) {
  PT_EAGER_DEBUG("Input data for updating H2D tensor:", data);
  launch_shapes.ds_tensors.push_back(dtensor);
  launch_shapes.patch_values.push_back(data);
  return;
}

int64_t UpdateDynamicTensorDSStack(
    torch::jit::IValue& iv_tensor,
    const std::vector<int64_t>& scalar_indexes,
    const std::vector<int64_t>& tensor_indexes,
    const std::vector<std::pair<int64_t, int64_t>>& mixed_indexes,
    std::shared_ptr<DynamicGraphMetaData> dmeta,
    const c10::SmallVector<int64_t, 8>& lookup_data) {
  int64_t stack_index = dmeta->ds_stack.size();
  dmeta->ds_stack.push_back(iv_tensor);

  // Mark the above created shape tensor with its corresponding symInts for
  // original stack.
  habana::graph::SymIntData STValue;
  STValue.values = scalar_indexes;
  STValue.lookup_data = lookup_data;
  dmeta->ds_tensor_to_scalar_map[stack_index] = STValue;
  dmeta->ds_tensor_to_tensor_map[stack_index] = tensor_indexes;
  dmeta->ds_mixed_map[stack_index] = mixed_indexes;
  PT_EAGER_DEBUG("Dynamic tensor inserted to stack at index:", stack_index);
  return stack_index;
}

int64_t CreateSTAndInsertToDSStack(
    const std::vector<int64_t>& st_size,
    const std::vector<int64_t>& scalar_indexes,
    const std::vector<int64_t>& tensor_indexes,
    const std::vector<std::pair<int64_t, int64_t>>& mixed_indexes,
    std::shared_ptr<DynamicGraphMetaData> dmeta,
    const c10::SmallVector<int64_t, 8>& lookup_data) {
  auto iv_st_tensor =
      torch::jit::IValue(createDynamicTensor(st_size, SHAPE_TENSOR));
  int64_t stack_index = UpdateDynamicTensorDSStack(
      iv_st_tensor,
      scalar_indexes,
      tensor_indexes,
      mixed_indexes,
      dmeta,
      lookup_data);
  return stack_index;
}

template <typename T>
int64_t CreateH2DAndInsertToDSStack(
    std::vector<int64_t>& values,
    std::vector<int64_t>& scalar_indexes,
    HostDataType dt_type,
    std::shared_ptr<DynamicGraphMetaData> dmeta) {
  std::vector<T> h2d_values(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    h2d_values[i] = static_cast<T>(values[i]);
  }
  at::Tensor h2d_tensor = createDynamicTensor(
      {static_cast<int64_t>(h2d_values.size())}, HOST_TO_DEVICE_TENSOR);
  SetH2DTensorHostData<T>(h2d_tensor, h2d_values, dt_type, true);

  auto iv_h2d_tensor = torch::jit::IValue(h2d_tensor);
  int64_t stack_index =
      UpdateDynamicTensorDSStack(iv_h2d_tensor, scalar_indexes, {}, {}, dmeta);
  return stack_index;
}

void DynamicOp::UpdateDynamicInputs(
    c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
    c10::SmallVectorImpl<habana::graph::SymIntData>& scalar_list,
    [[maybe_unused]] c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
    [[maybe_unused]] c10::SmallVectorImpl<
        std::vector<std::pair<int64_t, int64_t>>>& mixed_list,
    std::vector<c10::IValue>& orig_stack,
    LaunchDynamicShapes& launch_shapes) {
  HABANA_ASSERT(
      dtensor_list.size() == scalar_list.size(),
      "Dtensor and SymIntData count not matching");
  int64_t tensor_count = dtensor_list.size();
  for (int idx = 0; idx < tensor_count; idx++) {
    auto dtensor = dtensor_list[idx]->toTensor();
    SymIntData& st_values = scalar_list[idx];
    UpdateShapeTensorSize(dtensor, st_values.values, orig_stack, launch_shapes);
  }
}

bool RepeatOperatorDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* aten_repeat_node,
    torch::jit::Stack& org_stack,
    GraphInputIndexMap& org_stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  HABANA_ASSERT(2 == aten_repeat_node->inputs().size());
  auto v_repeat_shape = aten_repeat_node->inputs().at(1);
  static const auto hpu_repeat_symbol{
      c10::Symbol::fromQualString("hpu::repeat_ht")};
  auto graph{aten_repeat_node->owningGraph()};
  auto list_construct_node{v_repeat_shape->node()};

  // Step 1: Collect shape and scalar pos used in ListConstruct input node
  std::vector<int64_t> values;
  std::vector<int64_t> scalar_indexes;
  GetValuesAndScalarIndexesFromListConstruct(
      list_construct_node,
      org_stack,
      org_stack_index_map,
      values,
      scalar_indexes);
  auto expr_values = GetRangeInfoExprFromListConstruct(
      list_construct_node, org_stack_index_map, m_range_infos);
  // Step2: Create H2D tensor and insert to graph inputs.
  auto repeat_h2d_name =
      GetDynamicTensorName(v_repeat_shape->debugName(), HOST_TO_DEVICE_TENSOR);
  int64_t stack_index = CreateH2DAndInsertToDSStack<int32_t>(
      values, scalar_indexes, HostDataType::INT32_T, m_dmeta);
  m_range_infos->emplace_back(habana_helpers::RangeInfo(
      {}, {}, GetExprFromString(expr_values), "INVALID", -2));

  auto v_h2d_tensor = graph->addInput(repeat_h2d_name);

  // Step3: Register patching function and tensor lists
  std::vector<int64_t> dtensor_indexes{stack_index};
  InputPatchPair patch_info(
      &RepeatOperatorDS::UpdateDynamicInputs, dtensor_indexes);
  m_dmeta->ds_input_patching_list.push_back(patch_info);

  // Step4: Create hpu::repear_ht node and insert to the graph
  CreateAndInsertDynamicNodeToGraph(
      graph,
      aten_repeat_node,
      hpu_repeat_symbol,
      {aten_repeat_node->input(0), v_h2d_tensor},
      value_ivalue_map);

  return true;
}

void RepeatOperatorDS::UpdateDynamicInputs(
    c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
    c10::SmallVectorImpl<habana::graph::SymIntData>& scalar_idx_list,
    [[maybe_unused]] c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
    [[maybe_unused]] c10::SmallVectorImpl<
        std::vector<std::pair<int64_t, int64_t>>>& mixed_list,
    std::vector<c10::IValue>& orig_stack,
    LaunchDynamicShapes& launch_shapes) {
  HABANA_ASSERT(
      dtensor_list.size() == scalar_idx_list.size(),
      "Dtensor and SymIntData count not matching");
  HABANA_ASSERT(dtensor_list.size() == 1, "Tensor count should be 1");

  auto dtensor = dtensor_list[0]->toTensor();
  SymIntData& scalar_idx = scalar_idx_list[0];

  std::vector<int32_t> updated_h2d_data;
  for (size_t idx = 0; idx < scalar_idx.values.size(); ++idx) {
    auto stack_index = scalar_idx.values[idx];
    if (stack_index == LONG_MAX) {
      std::vector<int32_t> h2d_data = GetH2DTensorHostData<int32_t>(dtensor);
      std::reverse(h2d_data.begin(), h2d_data.end());
      updated_h2d_data.push_back(h2d_data[idx]);
    } else {
      updated_h2d_data.push_back(
          static_cast<int32_t>(GetSymintValue(orig_stack, stack_index)));
    }
  }

  std::reverse(updated_h2d_data.begin(), updated_h2d_data.end());
  std::vector<int64_t> cast_data(
      updated_h2d_data.begin(), updated_h2d_data.end());
  UpdateH2DPatchingData(dtensor, cast_data, launch_shapes);
}

bool TopkOperatorDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* aten_topk_node,
    torch::jit::Stack& org_stack,
    GraphInputIndexMap& org_stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  HABANA_ASSERT(5 == aten_topk_node->inputs().size());
  static const auto hpu_topk_symbol{c10::Symbol::fromQualString("hpu::topk")};
  auto v_k = aten_topk_node->inputs().at(1);
  auto graph{aten_topk_node->owningGraph()};

  // Step 1: Collect shape and scalar pos K node
  int64_t scalar_idx = LONG_MAX;
  int64_t act_value = 0;
  GetValueAndScalarIndexFromInput(
      v_k, org_stack, org_stack_index_map, act_value, scalar_idx);
  PT_EAGER_DEBUG("ST data:", act_value);

  // Step2: Create shape tensor and insert to graph inputs.
  auto k_st_name = GetDynamicTensorName(v_k->debugName(), SHAPE_TENSOR);
  int64_t stack_index =
      CreateSTAndInsertToDSStack({act_value}, {scalar_idx}, {}, {}, m_dmeta);
  auto v_st_tensor = graph->addInput(k_st_name);
  auto expr =
      GetRangeInfoExprFromInput(v_k, org_stack_index_map, m_range_infos);
  m_range_infos->emplace_back(habana_helpers::RangeInfo(
      {}, {}, GetExprFromString({expr}), "INVALID", -1));

  // Step3: Register patching function and tensor lists
  std::vector<int64_t> dtensor_indexes{stack_index};
  InputPatchPair patch_info(&DynamicOp::UpdateDynamicInputs, dtensor_indexes);
  m_dmeta->ds_input_patching_list.push_back(patch_info);

  // Step4: Create hpu::topk node and insert to the graph
  CreateAndInsertDynamicNodeToGraph(
      graph,
      aten_topk_node,
      hpu_topk_symbol,
      {aten_topk_node->input(0),
       v_st_tensor,
       aten_topk_node->input(2),
       aten_topk_node->input(3),
       aten_topk_node->input(4)},
      value_ivalue_map);

  return true;
}

// Dynamic shape (DS) support for select_scatter using shape tensor
bool SelectScatterOperatorDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* aten_select_scatter_node,
    torch::jit::Stack& org_stack,
    GraphInputIndexMap& org_stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  // 4 inputs in select_scatter
  // input: Tensor
  // src: Tensor
  // dim: Scalar
  // index: Scalar
  HABANA_ASSERT(4 == aten_select_scatter_node->inputs().size());
  static const auto hpu_select_scatter_symbol{c10::Symbol::fromQualString("hpu::select_scatter")};

  // 2 scalars: dim and index
  auto dim = aten_select_scatter_node->inputs().at(2);
  auto index = aten_select_scatter_node->inputs().at(3);
  auto graph{aten_select_scatter_node->owningGraph()};

  // Step 1: Collect shape and scalar for index and dim node
  int64_t index_idx = LONG_MAX;
  int64_t index_value = 0;
  GetValueAndScalarIndexFromInput(
      index, org_stack, org_stack_index_map, index_value, index_idx);
  PT_EAGER_DEBUG("ST index data:", index_value);

  int64_t dim_idx = LONG_MAX;
  int64_t dim_value = 0;
  GetValueAndScalarIndexFromInput(
      dim, org_stack, org_stack_index_map, dim_value, dim_idx);
  PT_EAGER_DEBUG("ST dim data:", dim_value);

  // Step2: Create shape tensor and insert to graph inputs.
  auto index_st_name = GetDynamicTensorName(index->debugName(), SHAPE_TENSOR);
  int64_t stack_index =
      CreateSTAndInsertToDSStack({index_value}, {index_idx}, {}, {}, m_dmeta);
  auto index_st_tensor = graph->addInput(index_st_name);
  auto expr =
      GetRangeInfoExprFromInput(index, org_stack_index_map, m_range_infos);
  m_range_infos->emplace_back(habana_helpers::RangeInfo(
      {}, {}, GetExprFromString({expr}), "INVALID", -1));

  auto dim_st_name = GetDynamicTensorName(dim->debugName(), SHAPE_TENSOR);
  int64_t dim_index =
      CreateSTAndInsertToDSStack({dim_value}, {dim_idx}, {}, {}, m_dmeta);
  auto dim_st_tensor = graph->addInput(dim_st_name);
  expr = GetRangeInfoExprFromInput(dim, org_stack_index_map, m_range_infos);
  m_range_infos->emplace_back(habana_helpers::RangeInfo(
      {}, {}, GetExprFromString({expr}), "INVALID", -1));

  // Step3: Register patching function and tensor lists
  std::vector<int64_t> dtensor_indexes{stack_index, dim_index};
  InputPatchPair patch_info(&DynamicOp::UpdateDynamicInputs, dtensor_indexes);
  m_dmeta->ds_input_patching_list.push_back(patch_info);

  // Step4: Create hpu::select_scatter node and insert to the graph
  CreateAndInsertDynamicNodeToGraph(
      graph,
      aten_select_scatter_node,
      hpu_select_scatter_symbol,
      {aten_select_scatter_node->input(0),
       aten_select_scatter_node->input(1),
       dim_st_tensor,
       index_st_tensor},
      value_ivalue_map);

  return true;
}

bool SliceOperatorDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* slice_node,
    torch::jit::Stack& in_stack,
    GraphInputIndexMap& org_stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  HABANA_ASSERT(6 == slice_node->inputs().size());
  auto graph{slice_node->owningGraph()};
  auto in_tensors = getInputTensors(slice_node, value_ivalue_map);

  // Step 1: Collect shape and scalar pos used in ListConstruct input node
  static const auto list_construct_symbol{
      c10::Symbol::fromQualString("prim::ListConstruct")};
  auto v_slice_shape = slice_node->inputs().at(5);
  HABANA_ASSERT(
      v_slice_shape->node()->kind() == list_construct_symbol,
      "Slice input is not a ListConstruct, it is: ",
      v_slice_shape->node()->kind().toQualString());
  auto list_construct_node{v_slice_shape->node()};

  std::vector<int64_t> self_size;
  std::vector<int64_t> scalar_indexes;
  GetValuesAndScalarIndexesFromListConstruct(
      list_construct_node,
      in_stack,
      org_stack_index_map,
      self_size,
      scalar_indexes);

  std::vector<std::pair<int64_t, int64_t>> mixed_indexes;
  std::vector<std::pair<int64_t, int64_t>> mixed_scalar_indexes;
  for (size_t i = 0; i < self_size.size(); i++)
    mixed_scalar_indexes.push_back(
        std::make_pair(scalar_indexes[i], self_size[i]));

  // get Dim
  int64_t dim_idx = LONG_MAX, start_idx = LONG_MAX, end_idx = LONG_MAX,
          step_idx = LONG_MAX;
  int64_t dim = 0, start = 0, end = 0, step = 0;
  GetValueAndScalarIndexFromInput(
      slice_node->inputs().at(1), in_stack, org_stack_index_map, dim, dim_idx);
  auto dim_expr = GetRangeInfoExprFromInput(
      slice_node->inputs().at(1), org_stack_index_map, m_range_infos);
  // get start
  GetValueAndScalarIndexFromInput(
      slice_node->inputs().at(2),
      in_stack,
      org_stack_index_map,
      start,
      start_idx);
  auto start_expr = GetRangeInfoExprFromInput(
      slice_node->inputs().at(2), org_stack_index_map, m_range_infos);
  // get end
  GetValueAndScalarIndexFromInput(
      slice_node->inputs().at(3), in_stack, org_stack_index_map, end, end_idx);
  auto end_expr = GetRangeInfoExprFromInput(
      slice_node->inputs().at(2), org_stack_index_map, m_range_infos);
  // get step
  GetValueAndScalarIndexFromInput(
      slice_node->inputs().at(4),
      in_stack,
      org_stack_index_map,
      step,
      step_idx);
  auto step_expr = GetRangeInfoExprFromInput(
      slice_node->inputs().at(3), org_stack_index_map, m_range_infos);

  // capture actual values
  mixed_indexes.push_back(std::make_pair(dim_idx, dim));
  mixed_indexes.push_back(std::make_pair(start_idx, start));
  mixed_indexes.push_back(std::make_pair(end_idx, end));
  mixed_indexes.push_back(std::make_pair(step_idx, step));

  dim = at::maybe_wrap_dim(dim, self_size.size(), /*wrap_scalar=*/true);
  end = self_size[dim] < end ? self_size[dim] : end;
  auto shape =
      SliceOperator::compute_output_shape(self_size, dim, start, end, step);

  // create shape tesnor
  auto slice_st_name = GetDynamicTensorName(
      slice_node->output()->debugName() + "0", SHAPE_TENSOR);
  std::vector<int64_t> tensor_indexes;
  {
    auto slice_input_0_val = slice_node->inputs().at(0);
    auto in_0_name = slice_input_0_val->debugName();
    if (org_stack_index_map.count(in_0_name)) {
      auto input_0_idx = static_cast<int64_t>(org_stack_index_map[in_0_name]);
      tensor_indexes.push_back(input_0_idx);
    }
  }
  int64_t st_stack_index = CreateSTAndInsertToDSStack(
      shape.at(0), scalar_indexes, tensor_indexes, mixed_indexes, m_dmeta);
  auto symbol_outputshape = c10::Symbol::attr("output_shapes");
  if (slice_node->hasAttribute(symbol_outputshape)) {
    std::string original = slice_node->s(symbol_outputshape);
    std::string modified = original.substr(1, original.length() - 2);
    m_range_infos->emplace_back(
        habana_helpers::RangeInfo({}, {}, modified, "INVALID", -1));
    PT_DYNAMIC_SHAPE_DEBUG("Adding ST ranges shape", modified);
  } else {
    PT_DYNAMIC_SHAPE_WARN(
        "ERROR: Not adding ST ranges output_shapes attribute null");
  }
  auto gip_slice_st_tensor = graph->addInput(slice_st_name);
  std::vector<int64_t> dtensor_indexes{st_stack_index};

  // H2D tensor
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_SLICE)) {
    // H2D value preperation
    std::vector<uint64_t> host_params{
        static_cast<uint64_t>(self_size.size()), 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
    std::vector<std::string> expr_sizes{
        std::to_string(self_size.size()),
        "1",
        "1",
        "1",
        "1",
        "1",
        "0",
        "0",
        "0",
        "0",
        "0"};
    int index = self_size.size() - dim;
    host_params[index] = step;
    expr_sizes[index] = step_expr;
    host_params[index + 5] = start;
    expr_sizes[index + 5] = start_expr;
    auto slice_h2d_name = GetDynamicTensorName(
        slice_node->output()->debugName() + "1", HOST_TO_DEVICE_TENSOR);
    at::Tensor h2d_tensor = createDynamicTensor(
        {static_cast<int64_t>(host_params.size()) * 2}, HOST_TO_DEVICE_TENSOR);
    SetH2DTensorHostData<uint64_t>(
        h2d_tensor, host_params, HostDataType::UINT64_T, false);
    m_range_infos->emplace_back(habana_helpers::RangeInfo(
        {}, {}, GetExprFromString(expr_sizes), "INVALID", -2));
    auto slice_h2d_tensor = torch::jit::IValue(h2d_tensor);
    int64_t h2d_stack_index = UpdateDynamicTensorDSStack(
        slice_h2d_tensor, {}, {}, mixed_scalar_indexes, m_dmeta);
    auto gip_slice_h2d_tensor = graph->addInput(slice_h2d_name);
    dtensor_indexes.push_back(h2d_stack_index);
    // Create hpu::slice_ht node and insert to the graph
    static const auto hpu_slice_ht_symbol{
        c10::Symbol::fromQualString("hpu::slice_ht")};
    CreateAndInsertDynamicNodeToGraph(
        graph,
        slice_node,
        hpu_slice_ht_symbol,
        {slice_node->input(0), gip_slice_st_tensor, gip_slice_h2d_tensor},
        value_ivalue_map);
  } else {
    auto dims = self_size.size();
    std::vector<int64_t> step_vec(dims, 1);
    std::vector<std::string> step_expr_vec(dims, "1");
    step_vec[dim] = step;
    step_expr_vec[dim] = step_expr;
    // create shape tesnor
    auto slice_st_name_1 = GetDynamicTensorName(
        slice_node->output()->debugName() + "1", SHAPE_TENSOR);
    int64_t st_stack_index_1 =
        CreateSTAndInsertToDSStack(step_vec, {}, {}, mixed_indexes, m_dmeta);
    m_range_infos->emplace_back(habana_helpers::RangeInfo(
        {}, {}, GetExprFromString({step_expr_vec}), "INVALID", -1));
    auto gip_slice_st_tensor_1 = graph->addInput(slice_st_name_1);
    dtensor_indexes.push_back(st_stack_index_1);
    std::vector<int64_t> start_vec(dims, 0);
    std::vector<std::string> start_expr_vec(dims, "0");
    start_vec[dim] = start;
    start_expr_vec[dim] = start_expr;
    // create shape tesnor
    auto slice_st_name_2 = GetDynamicTensorName(
        slice_node->output()->debugName() + "2", SHAPE_TENSOR);
    int64_t st_stack_index_2 =
        CreateSTAndInsertToDSStack(start_vec, {}, {}, mixed_indexes, m_dmeta);
    m_range_infos->emplace_back(habana_helpers::RangeInfo(
        {}, {}, GetExprFromString({start_expr_vec}), "INVALID", -1));
    auto gip_slice_st_tensor_2 = graph->addInput(slice_st_name_2);
    dtensor_indexes.push_back(st_stack_index_2);
    // Create hpu::slice_ht node and insert to the graph
    static const auto hpu_slice_ht_symbol{
        c10::Symbol::fromQualString("hpu::slice")};
    CreateAndInsertDynamicNodeToGraph(
        graph,
        slice_node,
        hpu_slice_ht_symbol,
        {slice_node->input(0),
         gip_slice_st_tensor,
         gip_slice_st_tensor_1,
         gip_slice_st_tensor_2},
        value_ivalue_map);
  }
  InputPatchPair patch_info(
      &SliceOperatorDS::UpdateDynamicInputs, dtensor_indexes);
  m_dmeta->ds_input_patching_list.push_back(patch_info);
  return true;
}

void SliceOperatorDS::UpdateDynamicInputs(
    c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
    [[maybe_unused]] c10::SmallVectorImpl<habana::graph::SymIntData>&
        scalar_idx_list,
    [[maybe_unused]] c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
    [[maybe_unused]] c10::SmallVectorImpl<
        std::vector<std::pair<int64_t, int64_t>>>& mixed_list,
    std::vector<c10::IValue>& orig_stack,
    LaunchDynamicShapes& launch_shapes) {
  // shape Tensor
  auto dtensorST = dtensor_list[0]->toTensor();
  std::vector<int64_t> self_size;
  launch_shapes.ds_tensors.push_back(dtensorST);
  for (size_t i = 0; i < mixed_list[1].size(); i++) {
    if (mixed_list[1].at(i).first == LONG_MAX)
      self_size.push_back(mixed_list[1].at(i).second);
    else
      self_size.push_back(
          GetSymintValue(orig_stack, mixed_list[1].at(i).first));
  }

  // tensor sizes
  {
    // patch Dim
    int64_t dim = 0;
    if (mixed_list[0].at(0).first == LONG_MAX ||
        mixed_list[0].at(0).first < 0) {
      dim = mixed_list[0].at(0).second;
    } else {
      dim = GetSymintValue(orig_stack, mixed_list[0].at(0).first);
    }
    // patch Start
    int64_t start = 0;
    if (mixed_list[0].at(1).first == LONG_MAX ||
        mixed_list[0].at(1).first < 0) {
      start = mixed_list[0].at(1).second;
    } else {
      start = GetSymintValue(orig_stack, mixed_list[0].at(1).first);
    }
    // patch end
    int64_t end = 0;
    if (mixed_list[0].at(2).first == LONG_MAX ||
        mixed_list[0].at(2).first < 0) {
      end = mixed_list[0].at(2).second;
    } else {
      end = GetSymintValue(orig_stack, mixed_list[0].at(2).first);
    }
    // patch step
    int64_t step = 0;
    if (mixed_list[0].at(3).first == LONG_MAX) {
      step = mixed_list[0].at(3).second;
    } else {
      step = GetSymintValue(orig_stack, mixed_list[0].at(3).first);
    }

    dim = at::maybe_wrap_dim(dim, self_size.size(), /*wrap_scalar=*/true);
    end = self_size[dim] < end ? self_size[dim] : end;
    auto shape =
        SliceOperator::compute_output_shape(self_size, dim, start, end, step);
    launch_shapes.patch_values.push_back(shape.at(0));
    // H2D patching
    auto dtensorH2D = dtensor_list[1]->toTensor();
    std::vector<uint64_t> host_params{
        static_cast<uint64_t>(self_size.size()), 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
    int index = self_size.size() - dim;
    host_params[index] = step;
    host_params[index + 5] = start;
    std::vector<int64_t> cast_data(host_params.begin(), host_params.end());
    UpdateH2DPatchingData(dtensorH2D, cast_data, launch_shapes);
  }
}

bool ExapndOperatorDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* node,
    torch::jit::Stack& stack,
    GraphInputIndexMap& stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> dmeta) {
  static const auto hpu_expand_ds_symbol{
      c10::Symbol::fromQualString("hpu::expand_ds")};
  torch::jit::Graph* graph = node->owningGraph();
  torch::jit::Value* sizes = node->input(1);
  std::vector<int64_t> values;
  std::vector<int64_t> scalar_ids;
  GetValuesAndScalarIndexesFromListConstruct(
      sizes->node(), stack, stack_index_map, values, scalar_ids);
  CreateAndInsertDynamicNodeToGraph(
      node->owningGraph(),
      node,
      hpu_expand_ds_symbol,
      {node->inputs()},
      value_ivalue_map)
      ->replaceInput(1, graph->addInput());

  at::IntArrayRef self_sizes =
      value_ivalue_map[node->input(0)]->toTensor().sizes();

  for (size_t i{}; i < values.size(); ++i)
    if (values[i] == -1)
      values[i] = self_sizes[i];

  std::vector<int64_t> dtensor_indexes{CreateSTAndInsertToDSStack(
      values, scalar_ids, {}, {}, dmeta, {values.begin(), values.end()})};

  auto symbol_outputshape = c10::Symbol::attr("output_shapes");
  if (node->hasAttribute(symbol_outputshape)) {
    std::string original = node->s(symbol_outputshape);
    std::string modified = original.substr(1, original.length() - 2);
    m_range_infos->emplace_back(
        habana_helpers::RangeInfo({}, {}, modified, "INVALID", -1));
    PT_DYNAMIC_SHAPE_DEBUG("Adding ST ranges shape", modified);
  } else {
    PT_DYNAMIC_SHAPE_WARN(
        "ERROR: Not adding ST ranges output_shapes attribute null");
  }

  InputPatchPair patch_info(
      &ExapndOperatorDS::UpdateDynamicInputs, dtensor_indexes);
  dmeta->ds_input_patching_list.push_back(patch_info);
  return true;
}

void ExapndOperatorDS::UpdateDynamicInputs(
    c10::SmallVectorImpl<at::IValue*>& ivals,
    c10::SmallVectorImpl<SymIntData>& scalars,
    [[maybe_unused]] c10::SmallVectorImpl<std::vector<int64_t>>& temp_unused,
    [[maybe_unused]] c10::SmallVectorImpl<
        std::vector<std::pair<int64_t, int64_t>>>& mixed_list,
    std::vector<at::IValue>& stack,
    LaunchDynamicShapes& launch_shapes) {
  at::IntArrayRef values = scalars[0].values;
  std::vector<int64_t> sizes(values.size(), 1);
  for (size_t i{}; i < values.size(); ++i) {
    bool isNegativeOrMaxLong = values[i] == -1 || values[i] == LONG_MAX;
    sizes[i] = isNegativeOrMaxLong ? scalars[0].lookup_data[i]
                                   : stack[values[i]].toInt();
  }
  launch_shapes.ds_tensors.push_back(ivals[0]->toTensor());
  launch_shapes.patch_values.push_back(sizes);
}

// Dynamic shape (DS) support for slice_scatter using shape tensor
// Use hpu::slice_scatter_ds op for supporting slice_scatter DS
// hpu::slice_scatter_ds is same as hpu::slice_insert_ds
// but hpu::slice_insert_ds cannot be used directly to avoid graph
// loops in case of inplace ops.
bool SliceScatterOperatorDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* aten_slice_scatter_node,
    torch::jit::Stack& org_stack,
    GraphInputIndexMap& org_stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  // 6 inputs in slice_scatter
  // input: Tensor
  // src: Tensor
  // dim: Scalar
  // start: Scalar
  // end: Scalar
  // step: Scalar

  HABANA_ASSERT(6 == aten_slice_scatter_node->inputs().size());
  static const auto hpu_slice_scatter_symbol{c10::Symbol::fromQualString("hpu::slice_scatter_ds")};

  at::IntArrayRef self_sizes = value_ivalue_map[aten_slice_scatter_node->input(0)]->toTensor().sizes();

  // 4 scalars: dim, start, end, step
  auto dim = aten_slice_scatter_node->inputs().at(2);
  auto start = aten_slice_scatter_node->inputs().at(3);
  auto step = aten_slice_scatter_node->inputs().at(5);
  auto graph{aten_slice_scatter_node->owningGraph()};

  // Step 1: Collect shape and scalar for step, end, start and dim node
  int64_t step_idx = LONG_MAX;
  int64_t step_value = 0;
  GetValueAndScalarIndexFromInput(
      step, org_stack, org_stack_index_map, step_value, step_idx);
  auto step_expr =
      GetRangeInfoExprFromInput(step, org_stack_index_map, m_range_infos);
  PT_EAGER_DEBUG("ST step data:", step_value);

  int64_t start_idx = LONG_MAX;
  int64_t start_value = 0;
  GetValueAndScalarIndexFromInput(
      start, org_stack, org_stack_index_map, start_value, start_idx);
  auto start_expr =
      GetRangeInfoExprFromInput(step, org_stack_index_map, m_range_infos);
  PT_EAGER_DEBUG("ST start data:", start_value);

  int64_t dim_idx = LONG_MAX;
  int64_t dim_value = 0;
  GetValueAndScalarIndexFromInput(
      dim, org_stack, org_stack_index_map, dim_value, dim_idx);
  auto dim_expr =
      GetRangeInfoExprFromInput(step, org_stack_index_map, m_range_infos);
  PT_EAGER_DEBUG("ST dim data:", dim_value);

  // Handle negative dimension
  dim_value = at::maybe_wrap_dim(dim_value, self_sizes.size());

  // Handle negative dimension for start parameter
  if (start_value < 0) {
     start_value = at::maybe_wrap_dim(start_value, self_sizes[dim_value]);
     start_idx = LONG_MAX;
  }

  // slice_insert_ds requires start and step values for all dimensions
  // Default value of start is 0 {0, ..., 0, actual start at dim, 0, ..., 0}
  // Default value of step is 1 {1, ..., 1, actual step at dim, 1, ..., 1}
  std::vector<long> start_value_final_v1 = {};
  std::vector<long> step_value_final_v1 = {};
  std::vector<long> start_idx_final_v1 = {};
  std::vector<long> step_idx_final_v1 = {};

  // Set the initial default values
  for(int j=0; j<dim_value; ++j) {
     start_value_final_v1.push_back(0);
     step_value_final_v1.push_back(1);
     start_idx_final_v1.push_back(LONG_MAX);
     step_idx_final_v1.push_back(LONG_MAX);
  }
  // Set the actual value at dim
  start_value_final_v1.push_back(start_value);
  step_value_final_v1.push_back(step_value);
  start_idx_final_v1.push_back(start_idx);
  step_idx_final_v1.push_back(step_idx);
  // Set the remaining default values
  for(int j=dim_value; j<(int)self_sizes.size()-1; ++j) {
     start_value_final_v1.push_back(0);
     step_value_final_v1.push_back(1);
     start_idx_final_v1.push_back(LONG_MAX);
     step_idx_final_v1.push_back(LONG_MAX);
  }
  const std::vector<long>& start_value_final = start_value_final_v1;
  const std::vector<long>& step_value_final = step_value_final_v1;
  const std::vector<long>& start_idx_final = start_idx_final_v1;
  const std::vector<long>& step_idx_final = step_idx_final_v1;

  // Step2: Create shape tensor and insert to graph inputs.
  auto step_st_name = GetDynamicTensorName(step->debugName(), SHAPE_TENSOR);
  int64_t step_index =
      CreateSTAndInsertToDSStack(step_value_final, step_idx_final, {}, {}, m_dmeta);
  auto step_st_tensor = graph->addInput(step_st_name);
  m_range_infos->emplace_back(habana_helpers::RangeInfo(
      {}, {}, GetExprFromString({step_expr}), "INVALID", -1));

  auto start_st_name = GetDynamicTensorName(start->debugName(), SHAPE_TENSOR);
  int64_t start_index =
      CreateSTAndInsertToDSStack(start_value_final, start_idx_final, {}, {}, m_dmeta);
  auto start_st_tensor = graph->addInput(start_st_name);
  m_range_infos->emplace_back(habana_helpers::RangeInfo(
      {}, {}, GetExprFromString({start_expr}), "INVALID", -1));

  // Step3: Register patching function and tensor lists
  std::vector<int64_t> dtensor_indexes{step_index, start_index};
  InputPatchPair patch_info(&DynamicOp::UpdateDynamicInputs, dtensor_indexes);
  m_dmeta->ds_input_patching_list.push_back(patch_info);

  // Step4: Create hpu::slice_scatter node and insert to the graph
  CreateAndInsertDynamicNodeToGraph(
      graph,
      aten_slice_scatter_node,
      hpu_slice_scatter_symbol,
      {aten_slice_scatter_node->input(0),
       aten_slice_scatter_node->input(1),
       step_st_tensor,
       start_st_tensor},
      value_ivalue_map);

  return true;
}

habana::graph::RegisterDSOps& DSOpsRegistry() {
  static habana::graph::RegisterDSOps* Registry =
      new habana::graph::RegisterDSOps();
  return *Registry;
}

#define DSOP_MID_BACKEND(op, className) \
  add(#op, []() { return std::make_shared<className>(); })

static const auto& BasicDSOpsRegistry =
    habana::graph::DSOpsRegistry()
        .DSOP_MID_BACKEND(aten::view, ViewOperatorDS)
        .DSOP_MID_BACKEND(hpu::view_neg, ViewOperatorDS)
        .DSOP_MID_BACKEND(aten::_unsafe_view, ViewOperatorDS)
        .DSOP_MID_BACKEND(aten::expand, ExapndOperatorDS)
        .DSOP_MID_BACKEND(aten::arange, ArangeOperatorDS)
        .DSOP_MID_BACKEND(aten::repeat, RepeatOperatorDS)
        .DSOP_MID_BACKEND(aten::topk, TopkOperatorDS)
        .DSOP_MID_BACKEND(aten::as_strided, AsStridedOperatorDS)
        .DSOP_MID_BACKEND(hpu::strided_insert, StridedInsertOperatorDS)
        .DSOP_MID_BACKEND(aten::select_scatter, SelectScatterOperatorDS)
        .DSOP_MID_BACKEND(aten::slice_scatter, SliceScatterOperatorDS)
        .DSOP_MID_BACKEND(hpu::slice_ds, SliceOperatorDS)
        .DSOP_MID_BACKEND(aten::as_strided_scatter, AsStridedScatterOperatorDS)
        .DSOP_MID_BACKEND(hpu::as_strided_scatter, AsStridedScatterOperatorDS)
        .DSOP_MID_BACKEND(
            hpu::as_strided_scatter_orig,
            AsStridedScatterOperatorDS)
        .DSOP_MID_BACKEND(hpu::habana_randperm, RandpermGeneratorOperatorDS)
        .DSOP_MID_BACKEND(hpu::habana_rand, RandOperatorDS)
        .DSOP_MID_BACKEND(hpu::habana_randn, RandnOperatorDS)
        .DSOP_MID_BACKEND(hpu::habana_randint, RandintOperatorDS)
        .DSOP_MID_BACKEND(
            aten::full,
            FullOpDS) // we are adding original schema name
        .DSOP_MID_BACKEND(aten::empty, EmptyOpDS)
        .DSOP_MID_BACKEND(hpu::constant_pad_nd_ds, ConstantPad2dOperatorDS);
} // namespace graph
} // namespace habana
