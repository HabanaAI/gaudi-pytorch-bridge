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

#include "habana_eager/graph_dynamic_ops.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "habana_eager/graph_dynamic.h"

#include "habana_helpers/logging.h"

namespace habana {
namespace graph {

void GetValueAndScalarIndexFromInput(
    torch::jit::Value* input,
    torch::jit::Stack& in_stack,
    GraphInputIndexMap& org_stack_index_map,
    int64_t& value,
    int64_t& index) {
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
    value = static_cast<int64_t>(in_stack[index].toScalar().toInt());
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
      new_shape[idx] = GetSymintValue(orig_stack, stack_index);
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
    std::shared_ptr<DynamicGraphMetaData> dmeta) {
  int64_t stack_index = dmeta->ds_stack.size();
  dmeta->ds_stack.push_back(iv_tensor);

  // Mark the above created shape tensor with its corresponding symInts for
  // original stack.
  habana::graph::SymIntData STValue;
  STValue.values = scalar_indexes;
  dmeta->ds_tensor_to_scalar_map[stack_index] = STValue;
  dmeta->ds_tensor_to_tensor_map[stack_index] = tensor_indexes;
  PT_EAGER_DEBUG("Dynamic tensor inserted to stack at index:", stack_index);
  return stack_index;
}

int64_t CreateSTAndInsertToDSStack(
    const std::vector<int64_t>& st_size,
    const std::vector<int64_t>& scalar_indexes,
    const std::vector<int64_t>& tensor_indexes,
    std::shared_ptr<DynamicGraphMetaData> dmeta) {
  auto iv_st_tensor =
      torch::jit::IValue(createDynamicTensor(st_size, SHAPE_TENSOR));
  int64_t stack_index = UpdateDynamicTensorDSStack(
      iv_st_tensor, scalar_indexes, tensor_indexes, dmeta);
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
      UpdateDynamicTensorDSStack(iv_h2d_tensor, scalar_indexes, {}, dmeta);
  return stack_index;
}

void DynamicOp::UpdateDynamicInputs(
    c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
    c10::SmallVectorImpl<habana::graph::SymIntData>& scalar_list,
    [[maybe_unused]] c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
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

  // Step2: Create H2D tensor and insert to graph inputs.
  auto repeat_h2d_name =
      GetDynamicTensorName(v_repeat_shape->debugName(), HOST_TO_DEVICE_TENSOR);
  int64_t stack_index = CreateH2DAndInsertToDSStack<int32_t>(
      values, scalar_indexes, HostDataType::INT32_T, m_dmeta);
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
      CreateSTAndInsertToDSStack({act_value}, {scalar_idx}, {}, m_dmeta);
  auto v_st_tensor = graph->addInput(k_st_name);

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
      CreateSTAndInsertToDSStack({index_value}, {index_idx}, {}, m_dmeta);
  auto index_st_tensor = graph->addInput(index_st_name);

  auto dim_st_name = GetDynamicTensorName(dim->debugName(), SHAPE_TENSOR);
  int64_t dim_index =
      CreateSTAndInsertToDSStack({dim_value}, {dim_idx}, {}, m_dmeta);
  auto dim_st_tensor = graph->addInput(dim_st_name);

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

// Dynamic shape (DS) support for slice_scatter using shape tensor
bool SliceScatterOperatorDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* aten_slice_scatter_node,
    torch::jit::Stack& org_stack,
    GraphInputIndexMap& org_stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  // 4 inputs in slice_scatter
  // input: Tensor
  // src: Tensor
  // dim: Scalar
  // start: Scalar
  // end: Scalar
  // step: Scalar
  HABANA_ASSERT(6 == aten_slice_scatter_node->inputs().size());
  static const auto hpu_slice_scatter_symbol{c10::Symbol::fromQualString("hpu::slice_scatter")};

  // 4 scalars: dim, start, end, step
  auto dim = aten_slice_scatter_node->inputs().at(2);
  auto start = aten_slice_scatter_node->inputs().at(3);
  auto end = aten_slice_scatter_node->inputs().at(4);
  auto step = aten_slice_scatter_node->inputs().at(5);
  auto graph{aten_slice_scatter_node->owningGraph()};

  // Step 1: Collect shape and scalar for step, end, start and dim node
  int64_t step_idx = LONG_MAX;
  int64_t step_value = 0;
  GetValueAndScalarIndexFromInput(
      step, org_stack, org_stack_index_map, step_value, step_idx);
  PT_EAGER_DEBUG("ST step data:", step_value);

  int64_t end_idx = LONG_MAX;
  int64_t end_value = 0;
  GetValueAndScalarIndexFromInput(
      end, org_stack, org_stack_index_map, end_value, end_idx);
  PT_EAGER_DEBUG("ST end data:", end_value);

  int64_t start_idx = LONG_MAX;
  int64_t start_value = 0;
  GetValueAndScalarIndexFromInput(
      start, org_stack, org_stack_index_map, start_value, start_idx);
  PT_EAGER_DEBUG("ST start data:", start_value);

  int64_t dim_idx = LONG_MAX;
  int64_t dim_value = 0;
  GetValueAndScalarIndexFromInput(
      dim, org_stack, org_stack_index_map, dim_value, dim_idx);
  PT_EAGER_DEBUG("ST dim data:", dim_value);

  // Step2: Create shape tensor and insert to graph inputs.
  auto step_st_name = GetDynamicTensorName(step->debugName(), SHAPE_TENSOR);
  int64_t step_index =
      CreateSTAndInsertToDSStack({step_value}, {step_idx}, {}, m_dmeta);
  auto step_st_tensor = graph->addInput(step_st_name);

  auto end_st_name = GetDynamicTensorName(end->debugName(), SHAPE_TENSOR);
  int64_t end_index =
      CreateSTAndInsertToDSStack({end_value}, {end_idx}, {}, m_dmeta);
  auto end_st_tensor = graph->addInput(end_st_name);

  auto start_st_name = GetDynamicTensorName(start->debugName(), SHAPE_TENSOR);
  int64_t start_index =
      CreateSTAndInsertToDSStack({start_value}, {start_idx}, {}, m_dmeta);
  auto start_st_tensor = graph->addInput(start_st_name);

  auto dim_st_name = GetDynamicTensorName(dim->debugName(), SHAPE_TENSOR);
  int64_t dim_index =
      CreateSTAndInsertToDSStack({dim_value}, {dim_idx}, {}, m_dmeta);
  auto dim_st_tensor = graph->addInput(dim_st_name);

  // Step3: Register patching function and tensor lists
  std::vector<int64_t> dtensor_indexes{step_index, end_index, start_index, dim_index};
  InputPatchPair patch_info(&DynamicOp::UpdateDynamicInputs, dtensor_indexes);
  m_dmeta->ds_input_patching_list.push_back(patch_info);

  // Step4: Create hpu::slice_scatter node and insert to the graph
  CreateAndInsertDynamicNodeToGraph(
      graph,
      aten_slice_scatter_node,
      hpu_slice_scatter_symbol,
      {aten_slice_scatter_node->input(0),
       aten_slice_scatter_node->input(1),
       dim_st_tensor,
       start_st_tensor,
       end_st_tensor,
       step_st_tensor},
      value_ivalue_map);

  return true;
}

habana::graph::RegisterDSOps& DSOpsRegistry() {
  static habana::graph::RegisterDSOps* Registry =
      new habana::graph::RegisterDSOps();
  return *Registry;
}

#define DSOP_MID_BACKEND(className) \
  []() { return std::make_shared<className>(); }

static auto& BasicDSOpsRegistry =
    habana::graph::DSOpsRegistry()
        .add("aten::view", DSOP_MID_BACKEND(ViewOperatorDS))
        .add("hpu::view_neg", DSOP_MID_BACKEND(ViewOperatorDS))
        .add("aten::arange", DSOP_MID_BACKEND(ArangeOperatorDS))
        .add("aten::repeat", DSOP_MID_BACKEND(RepeatOperatorDS))
        .add("aten::topk", DSOP_MID_BACKEND(TopkOperatorDS))
        .add("aten::as_strided", DSOP_MID_BACKEND(AsStridedOperatorDS))
        .add("hpu::strided_insert", DSOP_MID_BACKEND(StridedInsertOperatorDS))
        .add("aten::slice_scatter", DSOP_MID_BACKEND(SliceScatterOperatorDS))
        .add(
            "hpu::habana_randperm",
            DSOP_MID_BACKEND(RandpermGeneratorOperatorDS))
        .add("aten::select_scatter", DSOP_MID_BACKEND(SelectScatterOperatorDS));
} // namespace graph
} // namespace habana
