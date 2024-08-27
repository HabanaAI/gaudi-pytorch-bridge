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

#include <torch/csrc/jit/ir/ir.h>
#include <memory>
#include <string>
#include <vector>
#include "backend/backend_meta.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "habana_eager/graph_dynamic.h"
#include "habana_eager/graph_dynamic_ops.h"

namespace habana {
namespace graph {
std::vector<std::string> string_tokenizer(std::string s) {
  std::vector<std::string> exprs;
  std::string modified_s = s.substr(1, s.length() - 2);
  std::stringstream ss(modified_s);
  std::string word;
  while (!ss.eof()) {
    std::getline(ss, word, ',');
    exprs.push_back(word);
  }
  return exprs;
}
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
  int64_t stack_index = CreateSTAndInsertToDSStack(
      inferred_st_sizes, scalar_indexes, {}, {}, m_dmeta);
  auto v_st_tensor = graph->addInput(view_st_name);
  auto symbol_outputshape = c10::Symbol::attr("output_shapes");
  if (aten_view_node->hasAttribute(symbol_outputshape)) {
    std::string original = aten_view_node->s(symbol_outputshape);
    std::string modified = original.substr(1, original.length() - 2);
    m_range_infos->emplace_back(
        habana_helpers::RangeInfo({}, {}, modified, "INVALID", -1));
    PT_DYNAMIC_SHAPE_DEBUG("Adding ST ranges shape", modified);
  } else {
    PT_DYNAMIC_SHAPE_WARN(
        "ERROR: Not adding ST ranges output_shapes attribute null");
  }
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

void ViewOperatorDS::UpdateDynamicInputs(
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
  auto dtensor = dtensor_list[0]->toTensor();
  SymIntData& st_values = scalar_idx_list[0];
  UpdateShapeTensorSize(dtensor, st_values.values, orig_stack, launch_shapes);
  auto new_shape = launch_shapes.patch_values.back();
  auto has_neg_size = false;
  for (auto val : new_shape) {
    if (val < 0) {
      has_neg_size = true;
      break;
    }
  }
  if (has_neg_size) {
    // if the sizes are negative, remove the tensor and sizes
    // The actual tensor and sizes are inserted in ResolveNegativeSizes
    launch_shapes.ds_tensors.pop_back();
    launch_shapes.patch_values.pop_back();
  }
}

void ViewOperatorDS::ResolveNegativeSizes(
    torch::jit::Node* node,
    std::unordered_map<CValPtr, torch::jit::IValue>& value_ivalue_map,
    LaunchDynamicShapes& launch_shapes) {
  auto view_st_value = node->inputs().at(1);
  auto view_out_value = node->outputs().at(0);
  auto cos_t_shapes = value_ivalue_map[view_out_value].toTensor().sizes().vec();
  auto ivsh_view_st = value_ivalue_map[view_st_value];
  launch_shapes.ds_tensors.push_back(ivsh_view_st.toTensor());
  launch_shapes.patch_values.push_back(cos_t_shapes);
}

int64_t get_arange_depth_ds_1(
    const float start,
    const float end,
    const float step) {
  TORCH_CHECK(step != 0.0, "step value can not be 0.");
  TORCH_CHECK(!((start > end) && (step > 0)), "step must be negative.");
  TORCH_CHECK(!((start < end) && (step < 0)), "step must be positive.");

  int64_t num_elements = static_cast<int64_t>(ceil((end - start) / step));
  return num_elements;
}

bool IssetToIntegralDType(
    std::vector<torch::jit::Value*> start_end_step,
    torch::jit::Stack& in_stack,
    GraphInputIndexMap& org_stack_index_map) {
  bool isitInger = true;
  for (auto& val : start_end_step) {
    if (val) {
      auto in_name = val->debugName();
      static const auto constant_symbol{
          c10::Symbol::fromQualString("prim::Constant")};
      if (val->node()->kind() == constant_symbol) {
        auto scalar_val = toIValue(val).value().toScalar();
        isitInger = isitInger && scalar_val.isIntegral(true);
      } else if (org_stack_index_map.count(in_name)) {
        auto index = static_cast<int64_t>(org_stack_index_map[in_name]);
        auto scalar_val = in_stack[index].toScalar();
        isitInger = isitInger && scalar_val.isIntegral(true);
      } else {
        HABANA_ASSERT(
            false,
            "IssetToIntegralDType Node input=",
            val,
            " is not a graph input or a prim::Constant");
      }
    }
  }
  return isitInger;
}

bool ArangeOperatorDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* aten_arange_node,
    torch::jit::Stack& org_stack,
    GraphInputIndexMap& org_stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  static const auto hpu_arange_symbol{
      c10::Symbol::fromQualString("hpu::arange")};
  auto arange_shape = aten_arange_node->inputs().at(0);
  auto graph{aten_arange_node->owningGraph()};
  torch::jit::Value* start = nullptr;
  torch::jit::Value* end = nullptr;
  torch::jit::Value* step = nullptr;
  torch::jit::Value* out_dtype_node_val = nullptr;
  torch::jit::Value* out_layout_node_val = nullptr;
  torch::jit::Value* out_device_node_val = nullptr;
  torch::jit::Value* out_pin_mem_node_val = nullptr;
  if (aten_arange_node->inputs().size() == 5) {
    end = aten_arange_node->inputs().at(0);
    out_dtype_node_val = aten_arange_node->inputs().at(1);
    out_layout_node_val = aten_arange_node->inputs().at(2);
    out_device_node_val = aten_arange_node->inputs().at(3);
    out_pin_mem_node_val = aten_arange_node->inputs().at(4);
  } else if (aten_arange_node->inputs().size() == 6) {
    start = aten_arange_node->inputs().at(0);
    end = aten_arange_node->inputs().at(1);
    out_dtype_node_val = aten_arange_node->inputs().at(2);
    out_layout_node_val = aten_arange_node->inputs().at(3);
    out_device_node_val = aten_arange_node->inputs().at(4);
    out_pin_mem_node_val = aten_arange_node->inputs().at(5);
  } else if (aten_arange_node->inputs().size() == 7) {
    start = aten_arange_node->inputs().at(0);
    end = aten_arange_node->inputs().at(1);
    step = aten_arange_node->inputs().at(2);
    out_dtype_node_val = aten_arange_node->inputs().at(3);
    out_layout_node_val = aten_arange_node->inputs().at(4);
    out_device_node_val = aten_arange_node->inputs().at(5);
    out_pin_mem_node_val = aten_arange_node->inputs().at(6);
  } else {
    return false;
  }

  bool setToIntegralDType =
      IssetToIntegralDType({start, end, step}, org_stack, org_stack_index_map);
  auto out_dtype_opt = toIValue(out_dtype_node_val).value();
  auto out_dtype = out_dtype_opt.toOptional<at::ScalarType>().value_or(
      setToIntegralDType ? at::ScalarType::Long
                         : torch::get_default_dtype_as_scalartype());
  std::vector<int64_t> dtensor_indexes;
  if (!(GET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR)))
    return false;
  if (!(c10::isFloatingType(out_dtype))) {
    std::vector<int64_t> scalar_indexes;
    std::vector<int> h2d_values;
    std::vector<std::string> h2d_expr;

    int64_t step_idx = LONG_MAX;
    int64_t step_value = 1;
    GetValueAndScalarIndexFromInput(
        step, org_stack, org_stack_index_map, step_value, step_idx);
    auto step_expr =
        GetRangeInfoExprFromInput(step, org_stack_index_map, m_range_infos);
    scalar_indexes.push_back(step_idx);
    h2d_values.push_back(static_cast<int>(step_value));
    h2d_expr.push_back(step_expr);

    int64_t end_idx = LONG_MAX;
    int64_t end_value = 1;
    GetValueAndScalarIndexFromInput(
        end, org_stack, org_stack_index_map, end_value, end_idx);
    auto end_expr =
        GetRangeInfoExprFromInput(end, org_stack_index_map, m_range_infos);
    scalar_indexes.push_back(end_idx);
    h2d_values.push_back(static_cast<int>(end_value));
    h2d_expr.push_back(end_expr);

    int64_t start_idx = LONG_MAX;
    int64_t start_value = 0;
    GetValueAndScalarIndexFromInput(
        start, org_stack, org_stack_index_map, start_value, start_idx);
    auto start_expr =
        GetRangeInfoExprFromInput(start, org_stack_index_map, m_range_infos);
    scalar_indexes.push_back(start_idx);
    h2d_values.push_back(static_cast<int>(start_value));
    h2d_expr.push_back(start_expr);

    // Step2: Create H2D tensor and insert to graph inputs.
    auto arange_h2d_name =
        GetDynamicTensorName(arange_shape->debugName(), HOST_TO_DEVICE_TENSOR);
    at::Tensor h2d_tensor = createDynamicTensor(
        {static_cast<int64_t>(h2d_values.size())},
        HOST_TO_DEVICE_TENSOR,
        out_dtype);

    SetH2DTensorHostData<int32_t>(
        h2d_tensor, h2d_values, HostDataType::INT32_T, true);
    auto iv_h2d_tensor = torch::jit::IValue(h2d_tensor);
    int64_t stack_index = UpdateDynamicTensorDSStack(
        iv_h2d_tensor, scalar_indexes, {}, {}, m_dmeta);
    auto v_h2d_tensor = graph->addInput(arange_h2d_name);
    dtensor_indexes.push_back(stack_index);
    m_range_infos->emplace_back(habana_helpers::RangeInfo(
        {}, {}, GetExprFromString(h2d_expr), "INVALID", -2));

    // Use actual reshape sizes and avoid sizes with dims "-1"
    auto out_tensors = getOutputTensers(aten_arange_node, value_ivalue_map);
    auto inferred_st_sizes = out_tensors[0].sizes().vec();

    // aliased floats into int64 because floats are from prim consts
    std::vector<std::pair<int64_t, int64_t>> mixed_indexes;
    mixed_indexes.push_back(std::make_pair(start_idx, start_value));
    mixed_indexes.push_back(std::make_pair(end_idx, end_value));
    mixed_indexes.push_back(std::make_pair(step_idx, step_value));

    // Step2: Create shape tensor and insert to graph inputs.
    auto arange_st_name = GetDynamicTensorName(end->debugName(), SHAPE_TENSOR);
    int64_t stack_index_2 = CreateSTAndInsertToDSStack(
        inferred_st_sizes, {}, {}, mixed_indexes, m_dmeta);
    auto arange_st_tensor = graph->addInput(arange_st_name);
    dtensor_indexes.push_back(stack_index_2);
    auto symbol_outputshape = c10::Symbol::attr("output_shapes");
    if (aten_arange_node->hasAttribute(symbol_outputshape)) {
      std::string original = aten_arange_node->s(symbol_outputshape);
      std::string modified = original.substr(1, original.length() - 2);
      m_range_infos->emplace_back(
          habana_helpers::RangeInfo({}, {}, modified, "INVALID", -1));
      PT_DYNAMIC_SHAPE_DEBUG("Adding ST ranges shape", modified);
    } else {
      PT_DYNAMIC_SHAPE_WARN(
          "ERROR: Not adding ST ranges output_shapes attribute null");
    }

    // Step4: Create hpu::arange_start_step node and insert to the graph
    CreateAndInsertDynamicNodeToGraph(
        graph,
        aten_arange_node,
        hpu_arange_symbol,
        {v_h2d_tensor,
         arange_st_tensor,
         out_dtype_node_val,
         out_layout_node_val,
         out_device_node_val,
         out_pin_mem_node_val},
        value_ivalue_map);
    // Step3: Register patching function and tensor lists
    InputPatchPair patch_info(
        &ArangeOperatorDS::UpdateDynamicInputs, dtensor_indexes);
    m_dmeta->ds_input_patching_list.push_back(patch_info);
    return true;
  } else {
    return false;
  }
}

void ArangeOperatorDS::UpdateDynamicInputs(
    c10::SmallVectorImpl<torch::jit::IValue*>& dtensor_list,
    [[maybe_unused]] c10::SmallVectorImpl<habana::graph::SymIntData>&
        scalar_idx_list,
    [[maybe_unused]] c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
    [[maybe_unused]] c10::SmallVectorImpl<
        std::vector<std::pair<int64_t, int64_t>>>& mixed_list,
    [[maybe_unused]] std::vector<c10::IValue>& orig_stack,
    LaunchDynamicShapes& launch_shapes) {
  // patch Start
  int64_t start = 0;
  auto symint_idx = mixed_list[1].at(0).first;
  auto symint_val = mixed_list[1].at(0).second;
  if ((symint_idx == LONG_MAX) || (symint_val < 0)) {
    start = symint_val;
  } else {
    start = GetSymintValue(orig_stack, symint_idx);
  }
  // patch End
  int64_t end = 0;
  symint_idx = mixed_list[1].at(1).first;
  symint_val = mixed_list[1].at(1).second;
  if ((symint_idx == LONG_MAX) || (symint_val < 0)) {
    end = symint_val;
  } else {
    end = GetSymintValue(orig_stack, symint_idx);
  }
  // patch Step
  int64_t step = 0;
  symint_idx = mixed_list[1].at(2).first;
  symint_val = mixed_list[1].at(2).second;
  if ((symint_idx == LONG_MAX) || (symint_val < 0)) {
    step = symint_val;
  } else {
    step = GetSymintValue(orig_stack, symint_idx);
  }

  // modify H2D Tensor
  auto dtensorH2D = dtensor_list[0]->toTensor();
  auto tmeta{habana::get_tensor_extra_meta(dtensorH2D)};
  if (tmeta->get_tensor_type() == HOST_TO_DEVICE_TENSOR) {
    habana::HostDataType h2d_dt_type = tmeta->get_host_dt_type();
    if (h2d_dt_type == habana::HostDataType::INT32_T) {
      std::vector<int32_t> updated_h2d_data;
      updated_h2d_data.push_back(static_cast<int>(start));
      updated_h2d_data.push_back(static_cast<int>(end));
      updated_h2d_data.push_back(static_cast<int>(step));
      std::vector<int64_t> cast_data(
          updated_h2d_data.begin(), updated_h2d_data.end());
      launch_shapes.ds_tensors.push_back(dtensorH2D);
      launch_shapes.patch_values.push_back(cast_data);

      // modify Shape Tensor
      auto dtensorST = dtensor_list[1]->toTensor();
      auto st_size = get_arange_depth_ds_1(start, end, step);
      launch_shapes.ds_tensors.push_back(dtensorST);
      launch_shapes.patch_values.push_back({st_size});
    }
  }
}

bool ConstantPad2dOperatorDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* aten_pad_node,
    torch::jit::Stack& org_stack,
    GraphInputIndexMap& org_stack_index_map,
    [[maybe_unused]] ValueIvalueMap& value_ivalue_map,
    [[maybe_unused]] std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  HABANA_ASSERT(4 == aten_pad_node->inputs().size());
  static const auto hpu_pad_symbol{
      c10::Symbol::fromQualString("hpu::constant_pad_nd")};
  static const auto list_construct_symbol{
      c10::Symbol::fromQualString("prim::ListConstruct")};
  auto graph{aten_pad_node->owningGraph()};
  auto v_pad_shape = aten_pad_node->inputs().at(1);
  HABANA_ASSERT(
      v_pad_shape->node()->kind() == list_construct_symbol,
      "Pad input is not a ListConstruct, it is: ",
      v_pad_shape->node()->kind().toQualString());
  auto list_construct_node{v_pad_shape->node()};

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

  std::vector<int64_t> pad_ht_vec(MAX_DIMENSIONS_NUM * 2, 0);
  std::vector<int64_t> scalar_indexes_ht(MAX_DIMENSIONS_NUM * 2, LONG_MAX);
  std::vector<std::string> pad_ht_vec_expr(MAX_DIMENSIONS_NUM * 2, "0");
  // assuming that "pad" has a pair of pad values corresponding to each
  // dim that needs to be padded.
  for (unsigned int i = 0; i < values.size() / 2; i++) {
    // Host tensor layout 1D - 10 elements:
    // pad_before[0]...pad_before[4], pad_after[0] ... pad_after[4] (for
    // dimensionality IFM less then 5 some elements not in use)
    pad_ht_vec[i] = values[2 * i];
    pad_ht_vec_expr[i] = expr_values[2 * i];
    pad_ht_vec[MAX_DIMENSIONS_NUM + i] = values[2 * i + 1];
    pad_ht_vec_expr[MAX_DIMENSIONS_NUM + i] = expr_values[2 * i + 1];
    scalar_indexes_ht[i] = scalar_indexes[2 * i];
    scalar_indexes_ht[MAX_DIMENSIONS_NUM + i] = scalar_indexes[2 * i + 1];
  }
  std::reverse(pad_ht_vec.begin(), pad_ht_vec.end());
  std::reverse(scalar_indexes_ht.begin(), scalar_indexes_ht.end());
  std::reverse(pad_ht_vec_expr.begin(), pad_ht_vec_expr.end());
  // Step2: Create H2D tensor and insert to graph inputs.

  auto padop_h2d_name =
      GetDynamicTensorName(v_pad_shape->debugName(), HOST_TO_DEVICE_TENSOR);

  int64_t stack_index_ht = CreateH2DAndInsertToDSStack<int32_t>(
      pad_ht_vec, scalar_indexes_ht, HostDataType::UINT32_T, m_dmeta);
  m_range_infos->emplace_back(habana_helpers::RangeInfo(
      {}, {}, GetExprFromString(pad_ht_vec_expr), "INVALID", -2));
  auto v_h2d_tensor = graph->addInput(padop_h2d_name);

  auto scalar_val = aten_pad_node->inputs().at(2);
  int64_t pad_val_idx = LONG_MAX;
  int64_t pad_val = 0;
  GetValueAndScalarIndexFromInput(
      scalar_val, org_stack, org_stack_index_map, pad_val, pad_val_idx);

  // Use actual reshape sizes and avoid sizes with dims "-1"
  auto out_tensors = getOutputTensers(aten_pad_node, value_ivalue_map);
  auto inferred_st_sizes = out_tensors[0].sizes().vec();
  // Step 1: Collect shape and scalar pos used in ListConstruct input node
  auto v_self_shape = aten_pad_node->inputs().at(3);
  HABANA_ASSERT(
      v_self_shape->node()->kind() == list_construct_symbol,
      "Self size input is not a ListConstruct, it is: ",
      v_self_shape->node()->kind().toQualString());
  auto list_construct_node1{v_self_shape->node()};

  std::vector<int64_t> self_size;
  std::vector<int64_t> scalar_indexes_pad;
  GetValuesAndScalarIndexesFromListConstruct(
      list_construct_node1,
      org_stack,
      org_stack_index_map,
      self_size,
      scalar_indexes_pad);
  std::vector<std::pair<int64_t, int64_t>> mixed_scalar_indexes;
  for (size_t i = 0; i < self_size.size(); i++) {
    mixed_scalar_indexes.push_back(
        std::make_pair(scalar_indexes_pad[i], self_size[i]));
  }
  // Step2: Create shape tensor and insert to graph inputs.
  auto pad_st_name =
      GetDynamicTensorName(v_pad_shape->debugName(), SHAPE_TENSOR);
  int64_t stack_index_st = CreateSTAndInsertToDSStack(
      inferred_st_sizes, {pad_val_idx}, {}, mixed_scalar_indexes, m_dmeta);
  auto symbol_outputshape = c10::Symbol::attr("output_shapes");
  if (aten_pad_node->hasAttribute(symbol_outputshape)) {
    std::string original = aten_pad_node->s(symbol_outputshape);
    std::string modified = original.substr(1, original.length() - 2);
    m_range_infos->emplace_back(
        habana_helpers::RangeInfo({}, {}, modified, "INVALID", -1));
    PT_DYNAMIC_SHAPE_DEBUG("Adding ST ranges shape", modified);
  } else {
    PT_DYNAMIC_SHAPE_WARN(
        "ERROR: Not adding ST ranges output_shapes attribute null");
  }
  auto pad_st_tensor = graph->addInput(pad_st_name);
  std::vector<int64_t> dtensor_indexes{stack_index_ht, stack_index_st};
  InputPatchPair patch_info(
      &ConstantPad2dOperatorDS::UpdateDynamicInputs, dtensor_indexes);
  m_dmeta->ds_input_patching_list.push_back(patch_info);

  // Step4: Create hpu::constant_pad_nd_ht node and insert to the graph
  CreateAndInsertDynamicNodeToGraph(
      graph,
      aten_pad_node,
      hpu_pad_symbol,
      {aten_pad_node->input(0),
       v_h2d_tensor,
       pad_st_tensor,
       aten_pad_node->input(2)},
      value_ivalue_map);
  return true;
}

void ConstantPad2dOperatorDS::UpdateDynamicInputs(
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
  auto dtensor = dtensor_list[0]->toTensor();

  std::vector<uint32_t> updated_h2d_data;
  SymIntData& scalar_idx = scalar_idx_list[0];
  for (unsigned int idx = 0; idx < scalar_idx.values.size(); idx++) {
    auto stack_index = scalar_idx.values[idx];
    if (stack_index == LONG_MAX || stack_index < 0) {
      std::vector<uint32_t> h2d_data = GetH2DTensorHostData<uint32_t>(dtensor);
      std::reverse(h2d_data.begin(), h2d_data.end());
      updated_h2d_data.push_back(h2d_data[idx]);
    } else {
      updated_h2d_data.push_back(
          static_cast<uint32_t>(GetSymintValue(orig_stack, stack_index)));
    }
  }
  std::reverse(updated_h2d_data.begin(), updated_h2d_data.end());
  // cast h2d data to int for correct ST calc (to account for negative pad array
  // values)
  std::vector<int32_t> pad_arr_int(
      updated_h2d_data.begin(), updated_h2d_data.end());
  std::vector<int64_t> cast_data(
      updated_h2d_data.begin(), updated_h2d_data.end());
  UpdateH2DPatchingData(dtensor, cast_data, launch_shapes);
  //  Update size ShapeTensor
  // get prev shape tensor saved
  auto dtensor1 = dtensor_list[1]->toTensor();
  std::vector<int64_t> shape;
  for (size_t i = 0; i < mixed_list[1].size(); i++) {
    if (mixed_list[1].at(i).first == LONG_MAX)
      shape.push_back(mixed_list[1].at(i).second);
    else {
      shape.push_back(GetSymintValue(orig_stack, mixed_list[1].at(i).first));
    }
  }
  auto ndim = (int64_t)shape.size(); // num dims of new tensor
  auto padlen = (int64_t)updated_h2d_data.size() / 2; // pad data from new H2D
  HABANA_ASSERT(
      padlen >= ndim, "pad array length should be >= ndims of input tensor");
  // update new shape using new input tensor shape and updated H2D pad data
  for (unsigned int i = 0; i < ndim; i++) {
    auto pad_start = pad_arr_int[i];
    auto pad_end = pad_arr_int[MAX_DIMENSIONS_NUM + i];
    shape[ndim - i - 1] += (pad_start + pad_end);
  }
  dtensor1.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
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
  auto in_tensors = getInputTensors(aten_as_strided_node, value_ivalue_map);
  auto self = in_tensors[0];
  auto as_strided_shape = aten_as_strided_node->inputs().at(1);
  auto as_strided_stride = aten_as_strided_node->inputs().at(2);
  auto as_strided_offset = aten_as_strided_node->inputs().at(3);
  auto graph{aten_as_strided_node->owningGraph()};
  auto shape_construct_node{as_strided_shape->node()};
  auto stride_construct_node{as_strided_stride->node()};
  static const auto hpu_as_strided_orig_symbol{
      c10::Symbol::fromQualString("hpu::strided_view_orig_ds_h2d")};
  static const auto hpu_as_strided_symbol{
      c10::Symbol::fromQualString("hpu::strided_view_ds_h2d")};

  std::vector<int64_t> tensor_indexes;
  int input_strided_view = -1;
  if (!m_input_new_base_sizes->empty()) {
    auto as_strided_node_ip_0 = aten_as_strided_node->input(0)->debugName();
    for (auto& input_base_sizes_pair : *m_input_new_base_sizes) {
      int64_t input_idx{input_base_sizes_pair.first};
      auto graph_inputs_idx = graph->inputs().at(input_idx)->debugName();
      if (strcmp(as_strided_node_ip_0.c_str(), graph_inputs_idx.c_str()) == 0) {
        tensor_indexes.push_back(input_idx);
        input_strided_view = input_idx;
        break;
      }
    }
  }

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
  int64_t stack_index = CreateSTAndInsertToDSStack(
      values_shapes, scalar_indexes_shape, tensor_indexes, {}, m_dmeta);
  auto v_st_sizes_tensor = graph->addInput(as_strided_shape_st_name);
  std::vector<int64_t> dtensor_indexes{stack_index};

  auto symbol_outputshape = c10::Symbol::attr("output_shapes");
  if (aten_as_strided_node->hasAttribute(symbol_outputshape)) {
    std::string original = aten_as_strided_node->s(symbol_outputshape);
    std::string modified = original.substr(1, original.length() - 2);
    m_range_infos->emplace_back(
        habana_helpers::RangeInfo({}, {}, modified, "INVALID", -1));
    PT_DYNAMIC_SHAPE_DEBUG("Adding ST ranges shape", modified);
  } else {
    PT_DYNAMIC_SHAPE_WARN(
        "ERROR: Not adding ST ranges output_shapes attribute null");
  }

  // Collect ST shape and symlnt pos using ListConstruct values for strides
  // format of filling [num_strides, offset, stride[0], stride[1]...]
  auto as_strided_stride_st_name = GetDynamicTensorName(
      as_strided_stride->debugName(), HOST_TO_DEVICE_TENSOR);
  std::vector<int64_t> scalar_indexes;
  std::vector<uint64_t> h2d_values;
  std::vector<std::string> h2d_expr;
  // Get offset value
  int64_t offset_idx = LONG_MAX;
  int64_t offset_value = 0;
  GetValueAndScalarIndexFromInput(
      as_strided_offset,
      in_stack,
      org_stack_index_map,
      offset_value,
      offset_idx);
  auto expr_offset = GetRangeInfoExprFromInput(
      as_strided_offset, org_stack_index_map, m_range_infos);
  // Fill the offset values
  scalar_indexes.push_back(offset_idx);
  h2d_values.push_back(static_cast<uint64_t>(offset_value));
  h2d_expr.push_back(expr_offset);
  // Get and fill strides values
  std::vector<int64_t> scalar_indexes_strides;
  std::vector<int64_t> values_strides;
  std::vector<std::string> expr_strides;
  auto self_strides = self.strides().vec();
  if (stride_construct_node->kind() == torch::jit::prim::Constant) {
    GetValuesAndScalarIndexesFromListConst(
        stride_construct_node, values_strides, scalar_indexes_strides);
    expr_strides = GetRangeInfoExprFromListConst(
        stride_construct_node, org_stack_index_map, m_range_infos);
  } else if (stride_construct_node->kind() == torch::jit::prim::ListConstruct) {
    GetValuesAndScalarIndexesFromListConstruct(
        stride_construct_node,
        in_stack,
        org_stack_index_map,
        values_strides,
        scalar_indexes_strides);
    expr_strides = GetRangeInfoExprFromListConstruct(
        stride_construct_node, org_stack_index_map, m_range_infos);
  }
  if (input_strided_view >= 0) {
    PT_DYNAMIC_SHAPE_DEBUG("Filling stride expression from input tensor");
    expr_strides =
        string_tokenizer(m_range_infos->at(input_strided_view).expr_strides);
  }
  // Fill the strides values in reverse order
  for (auto it = values_strides.rbegin(); it != values_strides.rend(); ++it) {
    h2d_values.push_back(static_cast<uint64_t>(*it));
  }
  for (auto it = expr_strides.rbegin(); it != expr_strides.rend(); ++it) {
    h2d_expr.push_back(*it);
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
    h2d_expr.push_back("0");
  }
  // Fill num_strides at 0 index
  scalar_indexes.insert(scalar_indexes.begin(), LONG_MAX);
  auto num_strides = (values_strides.size() == 0) ? 1 : values_strides.size();
  h2d_values.insert(h2d_values.begin(), static_cast<uint64_t>(num_strides));
  h2d_expr.insert(h2d_expr.begin(), std::to_string(num_strides));

  at::Tensor h2d_tensor_strides = createDynamicTensor(
      {static_cast<int64_t>(h2d_values.size()) * 2}, HOST_TO_DEVICE_TENSOR);
  SetH2DTensorHostData<uint64_t>(
      h2d_tensor_strides, h2d_values, HostDataType::UINT64_T, false);
  auto iv_st_strides_tensor = torch::jit::IValue(h2d_tensor_strides);
  int64_t stack_index_strides = UpdateDynamicTensorDSStack(
      iv_st_strides_tensor, scalar_indexes, tensor_indexes, {}, m_dmeta);
  auto v_st_strides_tensor = graph->addInput(as_strided_stride_st_name);
  dtensor_indexes.push_back(stack_index_strides);
  m_range_infos->emplace_back(habana_helpers::RangeInfo(
      {}, {}, GetExprFromString(h2d_expr), "INVALID", -2));

  // Due to InputView handling case the self tensor size is set to base tensor
  // in LaunchRecipe and it is set contiguous, Now since this base tensor goes
  // as self tensor in backend(ratio case or otherwise), hence we need to do
  // claculation the same way here and pass to backend. Identify if it is base
  // tensor view case and apply the same logic in frontend.
  if (!tensor_indexes.empty()) {
    auto base_sizes_to_set = habana::get_base_tensor_size(self);
    self_strides = habana_helpers::calculate_strides(base_sizes_to_set);
  }
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
        iv_st_offset_tensor,
        scalar_indexes_offset,
        tensor_indexes,
        {},
        m_dmeta);
    auto as_strided_offset_st_name =
        GetDynamicTensorName(as_strided_offset->debugName(), SHAPE_TENSOR);
    auto v_st_offset_tensor = graph->addInput(as_strided_offset_st_name);
    dtensor_indexes.push_back(stack_index_offset);
    m_range_infos->emplace_back(habana_helpers::RangeInfo(
        {}, {}, GetExprFromString({expr_offset}), "INVALID", -1));
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
    c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
    [[maybe_unused]] c10::SmallVectorImpl<
        std::vector<std::pair<int64_t, int64_t>>>& mixed_list,
    std::vector<c10::IValue>& orig_stack,
    LaunchDynamicShapes& launch_shapes) {
  HABANA_ASSERT(
      dtensor_list.size() == scalar_idx_list.size(),
      "Dtensor and SymIntData count not matching");
  HABANA_ASSERT(
      dtensor_list.size() == scalar_idx_list.size(),
      "Dtensor and SymIntData count not matching");

  // Stride H2D patching
  // Format of H2D [num_strides, offset, stride[n], stride[n-1], .., 0]
  auto dtensor = dtensor_list[1]->toTensor();
  std::vector<uint64_t> updated_h2d_data;
  if (!tensor_list[0].empty()) {
    // Patch from tensor sizes and strides
    auto stack_idx = tensor_list[0].at(0);
    auto stack_tensor = orig_stack[stack_idx].toTensor();
    auto* impl = stack_tensor.unsafeGetTensorImpl();
    std::vector<uint64_t> h2d_data = GetH2DTensorHostData<uint64_t>(dtensor);
    updated_h2d_data.reserve(h2d_data.size());
    // Update strides and offset
    auto offset = impl->storage_offset();
    {
      updated_h2d_data.push_back(static_cast<uint64_t>(offset));
      std::vector<int64_t> values_strides = impl->strides().vec();
      for (auto it = values_strides.rbegin(); it != values_strides.rend();
           ++it) {
        updated_h2d_data.push_back(static_cast<uint64_t>(*it));
      }
      size_t fill_dim = (SYN_MAX_TENSOR_DIM + 1) - values_strides.size();
      for (size_t i = 0; i < fill_dim; i++) {
        updated_h2d_data.push_back(static_cast<uint64_t>(0));
      }
      updated_h2d_data.insert(
          updated_h2d_data.begin(),
          static_cast<uint64_t>(values_strides.size()));
      // fill remaining half with same data
      for (size_t idx = 0; idx < (h2d_data.size() >> 1); idx++) {
        updated_h2d_data.push_back(updated_h2d_data[idx]);
      }
      std::vector<int64_t> cast_data(
          updated_h2d_data.begin(), updated_h2d_data.end());
      UpdateH2DPatchingData(dtensor, cast_data, launch_shapes);
    }
    // Update offset ShapeTensor if present
    if (dtensor_list.size() == 3) {
      auto dtensor = dtensor_list[2]->toTensor();
      c10::SmallVector<int64_t, NUM_TENSOR_DIMS> new_shape(1, offset);
      std::vector<int64_t> return_shapes(new_shape.begin(), new_shape.end());
      launch_shapes.ds_tensors.push_back(dtensor);
      launch_shapes.patch_values.push_back(return_shapes);
      PT_EAGER_DEBUG("Updated offset shape tensor size:", dtensor.sizes());
    }
    // Update size ShapeTensor
    {
      auto dtensor = dtensor_list[0]->toTensor();
      std::vector<int64_t> values_sizes = impl->sizes().vec();
      PT_EAGER_DEBUG("Output ShapeTensor updated size ", values_sizes);
      launch_shapes.ds_tensors.push_back(dtensor);
      launch_shapes.patch_values.push_back(values_sizes);
    }
  } else {
    SymIntData& scalar_idx = scalar_idx_list[1];
    for (size_t idx = 0; idx < scalar_idx.values.size(); idx++) {
      auto stack_index = scalar_idx.values[idx];
      if (stack_index == LONG_MAX) {
        std::vector<uint64_t> h2d_data =
            GetH2DTensorHostData<uint64_t>(dtensor);
        updated_h2d_data.push_back(h2d_data[idx]);
      } else {
        updated_h2d_data.push_back(
            static_cast<uint64_t>(GetSymintValue(orig_stack, stack_index)));
      }
    }

    std::vector<int64_t> cast_data(
        updated_h2d_data.begin(), updated_h2d_data.end());
    UpdateH2DPatchingData(dtensor, cast_data, launch_shapes);
    for (int i = dtensor_list.size() - 1; i >= 0; i--) {
      if (i == 1)
        continue;
      auto dtensor = dtensor_list[i]->toTensor();
      SymIntData st_values = scalar_idx_list[i];
      UpdateShapeTensorSize(
          dtensor, st_values.values, orig_stack, launch_shapes);
    }
  }
}

// Dynamic shape (DS) support for as_strided_scatter op
bool AsStridedScatterOperatorDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* as_strided_scatter_node,
    torch::jit::Stack& in_stack,
    GraphInputIndexMap& org_stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  // as_strided_scatter op has 4 or 5 inputs:
  // input: Tensor
  // src: Tensor
  // size: Tuple or ints
  // stride: Tuple or ints
  // storage_offset: int (optional)
  HABANA_ASSERT(4 == as_strided_scatter_node->inputs().size() ||
                5 == as_strided_scatter_node->inputs().size());

  auto in_tensors = getInputTensors(as_strided_scatter_node, value_ivalue_map);
  auto self = in_tensors[0];
  // size is the shape of output tensor. This is not handled by the dynamic op
  // as strided_insert op (which is internally used by as_strided_scatter) does
  // not need size.
  // Handle stride (index 3) and offset (index 4).
  auto as_strided_scatter_stride = as_strided_scatter_node->inputs().at(3);
  auto as_strided_scatter_offset = as_strided_scatter_node->inputs().at(4);

  // 2 paths / DS variants for as_strided_scatter op:
  // StridedRatio and Non-StridedRatio (normal) path
  // hpu::as_strided_scatter_orig does not have storage_offset separately
  // A common tensor stores both stride and offset values
  static const auto hpu_as_strided_scatter_orig_symbol{
      c10::Symbol::fromQualString("hpu::as_strided_scatter_orig")};
  // hpu::as_strided_scatter has storage_offset separately
  static const auto hpu_as_strided_scatter_symbol{
      c10::Symbol::fromQualString("hpu::as_strided_scatter")};

  auto stride_construct_node{as_strided_scatter_stride->node()};
  auto graph{as_strided_scatter_node->owningGraph()};
  std::vector<int64_t> dtensor_indexes;

  // Create ST in the following format
  // { Number of strides, Offset, Stride[0], Stride[1], ..., Stride[N] }
  auto as_strided_scatter_stride_st_name = GetDynamicTensorName(
      as_strided_scatter_stride->debugName(), HOST_TO_DEVICE_TENSOR);

  std::vector<int64_t> scalar_indexes;
  std::vector<uint64_t> h2d_values;
  std::vector<std::string> h2d_expr;

  // Get offset value and index and add it to h2d_values and scalar_indexes respectively
  int64_t offset_idx = LONG_MAX;
  int64_t offset_value = 0;
  GetValueAndScalarIndexFromInput(
      as_strided_scatter_offset,
      in_stack,
      org_stack_index_map,
      offset_value,
      offset_idx);
  auto expr_offset = GetRangeInfoExprFromInput(
      as_strided_scatter_offset, org_stack_index_map, m_range_infos);
  scalar_indexes.push_back(offset_idx);
  h2d_values.push_back(static_cast<uint64_t>(offset_value));
  h2d_expr.push_back(expr_offset);

  // Stride is a vector
  // There is a stride value for each dimension of input tensor
  std::vector<int64_t> scalar_indexes_strides;
  std::vector<int64_t> values_strides;
  std::vector<std::string> expr_strides;
  // Get stride value and index and add it to h2d_values and scalar_indexes respectively
  auto self_strides = self.strides().vec();
  GetValuesAndScalarIndexesFromListConstruct(
      stride_construct_node,
      in_stack,
      org_stack_index_map,
      values_strides,
      scalar_indexes_strides);
  expr_strides = GetRangeInfoExprFromListConstruct(
      stride_construct_node, org_stack_index_map, m_range_infos);
  // Fill the strides values in reverse order
  for (auto it = values_strides.rbegin(); it != values_strides.rend(); ++it) {
    h2d_values.push_back(static_cast<uint64_t>(*it));
  }
  for (auto it = expr_strides.rbegin(); it != expr_strides.rend(); ++it) {
    h2d_expr.push_back(*it);
  }
  // Insert strides indexes in reverse order
  for (auto it = scalar_indexes_strides.rbegin(); it != scalar_indexes_strides.rend(); ++it) {
    scalar_indexes.push_back(*it);
  }
  // Fill the remaining dimension stride values with 0 and index with LONG_MAX
  // SYN_MAX_TENSOR_DIM is the maximum number of dimensions supported in DS
  // If actual dimension is less, fill the remaining values with default value.
  size_t fill_dim = (SYN_MAX_TENSOR_DIM + 1) - values_strides.size();
  for (size_t i = 0; i < fill_dim; i++) {
    h2d_values.push_back(static_cast<uint64_t>(0));
    scalar_indexes.push_back(LONG_MAX);
    h2d_expr.push_back("0");
  }
  // Insert num_strides at 0 index (first value)
  // and LONG_MAX as corresponding index
  scalar_indexes.insert(scalar_indexes.begin(), LONG_MAX);
  h2d_values.insert(h2d_values.begin(), static_cast<uint64_t>(values_strides.size()));
  h2d_expr.insert(h2d_expr.begin(), std::to_string(values_strides.size()));
  // Create H2D tensor using h2d_values and scalar_indexes
  at::Tensor h2d_tensor_strides = createDynamicTensor(
      {static_cast<int64_t>(h2d_values.size()) * 2}, HOST_TO_DEVICE_TENSOR);
  SetH2DTensorHostData<uint64_t>(h2d_tensor_strides, h2d_values, HostDataType::UINT64_T, false);
  auto iv_st_strides_tensor = torch::jit::IValue(h2d_tensor_strides);
  int64_t stack_index_strides = UpdateDynamicTensorDSStack(
      iv_st_strides_tensor, scalar_indexes, {}, {}, m_dmeta);
  auto v_st_strides_tensor = graph->addInput(as_strided_scatter_stride_st_name);
  dtensor_indexes.push_back(stack_index_strides);
  m_range_infos->emplace_back(habana_helpers::RangeInfo(
      {}, {}, GetExprFromString(h2d_expr), "INVALID", -2));

  // There are two paths: StridedRatio path or normal path
  // Path 1: Normal path / Non-StridedRatio path
  if (IsStridedRatioUndefined(self_strides, values_strides)) {
    // Actual stride values are not an integral multiple of input tensor strides (view op)
    auto tmeta{get_tensor_extra_meta(h2d_tensor_strides)};
    tmeta->set_H2D_data_for_bucketing();
    // Create hpu::as_strided_scatter_orig node and insert to the graph
    // No seperate parameter for offset
    CreateAndInsertDynamicNodeToGraph(
        graph,
        as_strided_scatter_node,
        hpu_as_strided_scatter_orig_symbol,
        {as_strided_scatter_node->input(0),
         as_strided_scatter_node->input(1),
         v_st_strides_tensor},
        value_ivalue_map);
  } else {
    // Path 2: StridedRatio path
    // Actual stride values are integral multiple of input tensor strides (view op)
    // Pass offset as a separate parameter
    // Create Shape Tensor for offset
    std::vector<int64_t> values_offset;
    values_offset.push_back(offset_value);
    at::Tensor st_tensor_offset =
        createDynamicTensor(values_offset, SHAPE_TENSOR);
    auto tmeta_offset{get_tensor_extra_meta(st_tensor_offset)};
    // Mark this front end shape tensor as it does not need synapse tensor.
    // It carries stride_ratios info for BE lowering kernel.
    tmeta_offset->set_H2D_frontend_shape_tensor();

    // Calculate stride ratios of passed strides and strides of input tensor
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
        iv_st_offset_tensor, scalar_indexes_offset, {}, {}, m_dmeta);

    auto as_strided_scatter_offset_st_name =
        GetDynamicTensorName(as_strided_scatter_offset->debugName(), SHAPE_TENSOR);
    auto v_st_offset_tensor = graph->addInput(as_strided_scatter_offset_st_name);
    dtensor_indexes.push_back(stack_index_offset);
    m_range_infos->emplace_back(habana_helpers::RangeInfo(
        {}, {}, GetExprFromString({expr_offset}), "INVALID", -1));

    // Create hpu::as_strided_scatter node and insert to the graph
    // This has offset parameter
    CreateAndInsertDynamicNodeToGraph(
        graph,
        as_strided_scatter_node,
        hpu_as_strided_scatter_symbol,
        {as_strided_scatter_node->input(0),
         as_strided_scatter_node->input(1),
         v_st_strides_tensor,
         v_st_offset_tensor},
        value_ivalue_map);
  }

  InputPatchPair patch_info(
      &AsStridedScatterOperatorDS::UpdateDynamicInputs, dtensor_indexes);
  m_dmeta->ds_input_patching_list.push_back(patch_info);
  return true;
}

// Update DS input for as_strided_scatter DS op
void AsStridedScatterOperatorDS::UpdateDynamicInputs(
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

  // Stride H2D patching
  // Format of H2D tensor:
  // { Number of strides, Offset, Stride[0], Stride[1], ..., Stride[N] }
  auto dtensor = dtensor_list[0]->toTensor();
  SymIntData& scalar_idx = scalar_idx_list[0];

  std::vector<uint64_t> updated_h2d_data;
  for (size_t idx = 0; idx < scalar_idx.values.size(); ++idx) {
    auto stack_index = scalar_idx.values[idx];
    if (stack_index == LONG_MAX) {
      std::vector<uint64_t> h2d_data = GetH2DTensorHostData<uint64_t>(dtensor);
      updated_h2d_data.push_back(h2d_data[idx]);
    } else {
      updated_h2d_data.push_back(
          static_cast<uint64_t>(GetSymintValue(orig_stack, stack_index)));
    }
  }
  std::vector<int64_t> cast_data(
      updated_h2d_data.begin(), updated_h2d_data.end());
  UpdateH2DPatchingData(dtensor, cast_data, launch_shapes);

  // offset patching if StridedRatio path is taken
  if (dtensor_list.size() == 2) {
    auto dtensor_offset = dtensor_list[1]->toTensor();
    SymIntData st_values = scalar_idx_list[1];
    UpdateShapeTensorSize(
        dtensor_offset, st_values.values, orig_stack, launch_shapes);
  }
}

bool StridedInsertOperatorDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* strided_insert_node,
    torch::jit::Stack& in_stack,
    GraphInputIndexMap& org_stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  HABANA_ASSERT(4 == strided_insert_node->inputs().size());
  auto in_tensors = getInputTensors(strided_insert_node, value_ivalue_map);
  auto self = in_tensors[0];
  auto strided_insert_stride = strided_insert_node->inputs().at(2);
  auto strided_insert_offset = strided_insert_node->inputs().at(3);
  static const auto hpu_strided_insert_orig_symbol{
      c10::Symbol::fromQualString("hpu::strided_insert_orig_ds_h2d")};
  static const auto hpu_strided_insert_symbol{
      c10::Symbol::fromQualString("hpu::strided_insert_orig_ds")};
  auto stride_construct_node{strided_insert_stride->node()};
  auto graph{strided_insert_node->owningGraph()};
  std::vector<int64_t> dtensor_indexes;
  // Collect ST shape and symlnt pos using ListConstruct values for strides
  // format of filling [num_strides, offset, stride[0], stride[1]...]
  auto strided_insert_stride_st_name = GetDynamicTensorName(
      strided_insert_stride->debugName(), HOST_TO_DEVICE_TENSOR);
  std::vector<int64_t> scalar_indexes;
  std::vector<uint64_t> h2d_values;
  std::vector<std::string> h2d_expr;
  // Get offset value
  int64_t offset_idx = LONG_MAX;
  int64_t offset_value = 0;
  GetValueAndScalarIndexFromInput(
      strided_insert_offset,
      in_stack,
      org_stack_index_map,
      offset_value,
      offset_idx);
  auto expr_offset = GetRangeInfoExprFromInput(
      strided_insert_offset, org_stack_index_map, m_range_infos);
  // Fill the offset values
  scalar_indexes.push_back(offset_idx);
  h2d_values.push_back(static_cast<uint64_t>(offset_value));
  h2d_expr.push_back(expr_offset);
  // Get and fill strides values
  std::vector<int64_t> scalar_indexes_strides;
  std::vector<int64_t> values_strides;
  auto self_strides = self.strides().vec();
  std::vector<std::string> expr_strides;
  static const auto constant_symbol{
      c10::Symbol::fromQualString("prim::Constant")};
  if (stride_construct_node->kind() == constant_symbol) {
      GetValuesAndScalarIndexesFromListConst(
          stride_construct_node,
          values_strides,
          scalar_indexes_strides);
      expr_strides = GetRangeInfoExprFromListConst(
          stride_construct_node, org_stack_index_map, m_range_infos);
  } else {
      GetValuesAndScalarIndexesFromListConstruct(
          stride_construct_node,
          in_stack,
          org_stack_index_map,
          values_strides,
          scalar_indexes_strides);
      expr_strides = GetRangeInfoExprFromListConstruct(
          stride_construct_node, org_stack_index_map, m_range_infos);
  }
  // Fill the strides values in reverse order
  for (auto it = values_strides.rbegin(); it != values_strides.rend(); ++it) {
    h2d_values.push_back(static_cast<uint64_t>(*it));
  }
  for (auto it = expr_strides.rbegin(); it != expr_strides.rend(); ++it) {
    h2d_expr.push_back(*it);
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
    h2d_expr.push_back("0");
  }
  // Fill num_strides at 0 index
  scalar_indexes.insert(scalar_indexes.begin(), LONG_MAX);
  h2d_values.insert(
      h2d_values.begin(), static_cast<uint64_t>(values_strides.size()));
  h2d_expr.insert(h2d_expr.begin(), std::to_string(values_strides.size()));

  at::Tensor h2d_tensor_strides = createDynamicTensor(
      {static_cast<int64_t>(h2d_values.size()) * 2}, HOST_TO_DEVICE_TENSOR);
  SetH2DTensorHostData<uint64_t>(
      h2d_tensor_strides, h2d_values, HostDataType::UINT64_T, false);
  auto iv_st_strides_tensor = torch::jit::IValue(h2d_tensor_strides);
  int64_t stack_index_strides = UpdateDynamicTensorDSStack(
      iv_st_strides_tensor, scalar_indexes, {}, {}, m_dmeta);
  auto v_st_strides_tensor = graph->addInput(strided_insert_stride_st_name);
  dtensor_indexes.push_back(stack_index_strides);
  m_range_infos->emplace_back(habana_helpers::RangeInfo(
      {}, {}, GetExprFromString(h2d_expr), "INVALID", -2));

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
        iv_st_offset_tensor, scalar_indexes_offset, {}, {}, m_dmeta);
    auto strided_insert_offset_st_name =
        GetDynamicTensorName(strided_insert_offset->debugName(), SHAPE_TENSOR);
    auto v_st_offset_tensor = graph->addInput(strided_insert_offset_st_name);
    dtensor_indexes.push_back(stack_index_offset);
    m_range_infos->emplace_back(habana_helpers::RangeInfo(
        {}, {}, GetExprFromString({expr_offset}), "INVALID", -1));
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
    [[maybe_unused]] c10::SmallVectorImpl<std::vector<int64_t>>& tensor_list,
    [[maybe_unused]] c10::SmallVectorImpl<
        std::vector<std::pair<int64_t, int64_t>>>& mixed_list,
    std::vector<c10::IValue>& orig_stack,
    LaunchDynamicShapes& launch_shapes) {
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
  for (size_t idx = 0; idx < scalar_idx.values.size(); ++idx) {
    auto stack_index = scalar_idx.values[idx];
    if (stack_index == LONG_MAX) {
      std::vector<uint64_t> h2d_data = GetH2DTensorHostData<uint64_t>(dtensor);
      updated_h2d_data.push_back(h2d_data[idx]);
    } else {
      updated_h2d_data.push_back(
          static_cast<uint64_t>(GetSymintValue(orig_stack, stack_index)));
    }
  }
  std::vector<int64_t> cast_data(
      updated_h2d_data.begin(), updated_h2d_data.end());
  UpdateH2DPatchingData(dtensor, cast_data, launch_shapes);

  // offset patching in case stridedratio flow is used
  // Need to update the stride information also to be patched
  // in case of dynamic cache hit
  // Read the relevant data from updated_h2d_data and update the tmeta
  // @TODO - Check if need to recalculate the stride ratio also
  if (dtensor_list.size() == 2) {
    auto dtensor_offset = dtensor_list[1]->toTensor();
    SymIntData st_values = scalar_idx_list[1];
    UpdateShapeTensorSize(
        dtensor_offset, st_values.values, orig_stack, launch_shapes);
  }
}

void RandpermGeneratorOperatorDS::UpdateDynamicInputs(
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
  for (unsigned int idx = 0; idx < scalar_idx.values.size(); idx++) {
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

bool RandpermGeneratorOperatorDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* aten_randperm_node,
    torch::jit::Stack& org_stack,
    GraphInputIndexMap& org_stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  HABANA_ASSERT(6 == aten_randperm_node->inputs().size());
  static const auto hpu_arange_symbol{
      c10::Symbol::fromQualString("hpu::habana_randperm_ht")};
  // pos-0 : seed tensor, pos-1: 'n'
  auto arange_shape = aten_randperm_node->inputs().at(1);
  auto graph{aten_randperm_node->owningGraph()};
  // pos-0 : seed tensor, pos-1: 'n'
  auto end = aten_randperm_node->inputs().at(1);

  std::vector<int64_t> scalar_indexes;
  std::vector<int64_t> h2d_values;
  std::vector<std::string> h2d_expr;

  int64_t step_idx = LONG_MAX;
  int64_t step_value = 1;
  scalar_indexes.push_back(step_idx);
  h2d_values.push_back(static_cast<uint64_t>(step_value));
  h2d_expr.push_back("1");

  int64_t end_idx = LONG_MAX;
  int64_t end_value = 0;
  GetValueAndScalarIndexFromInput(
      end, org_stack, org_stack_index_map, end_value, end_idx);
  auto expr_end =
      GetRangeInfoExprFromInput(end, org_stack_index_map, m_range_infos);
  scalar_indexes.push_back(end_idx);
  h2d_values.push_back(static_cast<uint64_t>(end_value));
  h2d_expr.push_back(expr_end);

  int64_t start_idx = LONG_MAX;
  int64_t start_value = 0;
  scalar_indexes.push_back(start_idx);
  h2d_values.push_back(static_cast<uint64_t>(start_value));
  h2d_expr.push_back("0");

  // Step2: Create H2D tensor and insert to graph inputs.
  auto arange_h2d_name =
      GetDynamicTensorName(arange_shape->debugName(), HOST_TO_DEVICE_TENSOR);
  int64_t stack_index = CreateH2DAndInsertToDSStack<int32_t>(
      h2d_values, scalar_indexes, HostDataType::INT32_T, m_dmeta);
  auto v_h2d_tensor = graph->addInput(arange_h2d_name);
  m_range_infos->emplace_back(habana_helpers::RangeInfo(
      {}, {}, GetExprFromString(h2d_expr), "INVALID", -2));

  // Step3: Register patching function and tensor lists
  std::vector<int64_t> dtensor_indexes{stack_index};
  InputPatchPair patch_info(
      &RandpermGeneratorOperatorDS::UpdateDynamicInputs, dtensor_indexes);
  m_dmeta->ds_input_patching_list.push_back(patch_info);

  // Use actual reshape sizes and avoid sizes with dims "-1"
  auto out_tensors = getOutputTensers(aten_randperm_node, value_ivalue_map);
  auto inferred_st_sizes = out_tensors[0].sizes().vec();

  // Step2: Create shape tensor and insert to graph inputs.
  auto arange_st_name = GetDynamicTensorName(end->debugName(), SHAPE_TENSOR);
  int64_t stack_index_2 =
      CreateSTAndInsertToDSStack(inferred_st_sizes, {end_idx}, {}, {}, m_dmeta);
  auto symbol_outputshape = c10::Symbol::attr("output_shapes");
  if (aten_randperm_node->hasAttribute(symbol_outputshape)) {
    std::string original = aten_randperm_node->s(symbol_outputshape);
    std::string modified = original.substr(1, original.length() - 2);
    m_range_infos->emplace_back(
        habana_helpers::RangeInfo({}, {}, modified, "INVALID", -1));
    PT_DYNAMIC_SHAPE_DEBUG("Adding ST ranges shape", modified);
  } else {
    PT_DYNAMIC_SHAPE_WARN(
        "ERROR: Not adding ST ranges output_shapes attribute null");
  }

  auto arange_st_tensor = graph->addInput(arange_st_name);

  // Step3: Register patching function and tensor lists
  std::vector<int64_t> dtensor_indexes_2{stack_index_2};
  InputPatchPair patch_info_2(
      &DynamicOp::UpdateDynamicInputs, dtensor_indexes_2);
  m_dmeta->ds_input_patching_list.push_back(patch_info_2);
  // Step4: Create hpu::arange_start_step node and insert to the graph
  CreateAndInsertDynamicNodeToGraph(
      graph,
      aten_randperm_node,
      hpu_arange_symbol,
      {aten_randperm_node->input(0),
       v_h2d_tensor,
       arange_st_tensor,
       aten_randperm_node->input(2),
       aten_randperm_node->input(3),
       aten_randperm_node->input(4),
       aten_randperm_node->input(5)},
      value_ivalue_map);
  return true;
}

// Dynamic shape (DS) support for `rand` using shape tensor
bool RandOperatorDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* aten_rand_node,
    torch::jit::Stack& org_stack,
    GraphInputIndexMap& org_stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  // 6 inputs in rand
  HABANA_ASSERT(6 == aten_rand_node->inputs().size());
  // 1 scalars: size
  auto v_rand_shape = aten_rand_node->inputs().at(1);
  // fetching the list symbol for size
  static const auto list_construct_symbol{
      c10::Symbol::fromQualString("prim::ListConstruct")};
  HABANA_ASSERT(
      v_rand_shape->node()->kind() == list_construct_symbol,
      "rand input is not a ListConstruct, it is: ",
      v_rand_shape->node()->kind().toQualString());
  auto list_construct_node{v_rand_shape->node()};
  auto graph{aten_rand_node->owningGraph()};

  // Step 1: Collect shape and scalar pos used in ListConstruct input node
  std::vector<int64_t> st_size;
  std::vector<int64_t> scalar_indexes;
  GetValuesAndScalarIndexesFromListConstruct(
      list_construct_node,
      org_stack,
      org_stack_index_map,
      st_size,
      scalar_indexes);
  auto out_tensors = getOutputTensers(aten_rand_node, value_ivalue_map);
  auto inferred_st_sizes = out_tensors[0].sizes().vec();
  // Step2: Create shape tensor and insert to graph inputs.
  auto rand_st_name =
      GetDynamicTensorName(v_rand_shape->debugName(), SHAPE_TENSOR);
  int64_t stack_index = CreateSTAndInsertToDSStack(
      inferred_st_sizes, scalar_indexes, {}, {}, m_dmeta);
  auto rand_st_tensor = graph->addInput(rand_st_name);

  auto symbol_outputshape = c10::Symbol::attr("output_shapes");
  if (aten_rand_node->hasAttribute(symbol_outputshape)) {
    std::string original = aten_rand_node->s(symbol_outputshape);
    std::string modified = original.substr(1, original.length() - 2);
    m_range_infos->emplace_back(
        habana_helpers::RangeInfo({}, {}, modified, "INVALID", -1));
    PT_DYNAMIC_SHAPE_DEBUG("Adding ST ranges shape", modified);
  } else {
    PT_DYNAMIC_SHAPE_WARN(
        "ERROR: Not adding ST ranges output_shapes attribute null");
  }

  // Step3: Create hpu::rand_ds node and insert to the graph
  static const auto hpu_rand_symbol{
      c10::Symbol::fromQualString("hpu::habana_rand_st")};
  CreateAndInsertDynamicNodeToGraph(
      graph,
      aten_rand_node,
      hpu_rand_symbol,
      {aten_rand_node->inputs().at(0),
       rand_st_tensor,
       aten_rand_node->inputs().at(2),
       aten_rand_node->inputs().at(3),
       aten_rand_node->inputs().at(4),
       aten_rand_node->inputs().at(5)},
      value_ivalue_map);

  // Step4: Register patching function and tensor lists
  std::vector<int64_t> dtensor_indexes{stack_index};
  InputPatchPair patch_info(
      &RandOperatorDS::UpdateDynamicInputs, dtensor_indexes);
  m_dmeta->ds_input_patching_list.push_back(patch_info);
  return true;
}

void RandOperatorDS::UpdateDynamicInputs(
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
  auto dtensor = dtensor_list[0]->toTensor();
  SymIntData& scalar_idx = scalar_idx_list[0];
  UpdateShapeTensorSize(dtensor, scalar_idx.values, orig_stack, launch_shapes);
}

// Dynamic shape (DS) support for `rand` using shape tensor
bool RandnOperatorDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* aten_rand_node,
    torch::jit::Stack& org_stack,
    GraphInputIndexMap& org_stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  // 6 inputs in rand
  HABANA_ASSERT(6 == aten_rand_node->inputs().size());
  // 1 scalars: size
  auto v_rand_shape = aten_rand_node->inputs().at(1);
  // fetching the list symbol for size
  static const auto list_construct_symbol{
      c10::Symbol::fromQualString("prim::ListConstruct")};
  HABANA_ASSERT(
      v_rand_shape->node()->kind() == list_construct_symbol,
      "rand input is not a ListConstruct, it is: ",
      v_rand_shape->node()->kind().toQualString());
  auto list_construct_node{v_rand_shape->node()};
  auto graph{aten_rand_node->owningGraph()};

  // Step 1: Collect shape and scalar pos used in ListConstruct input node
  std::vector<int64_t> st_size;
  std::vector<int64_t> scalar_indexes;
  GetValuesAndScalarIndexesFromListConstruct(
      list_construct_node,
      org_stack,
      org_stack_index_map,
      st_size,
      scalar_indexes);
  auto out_tensors = getOutputTensers(aten_rand_node, value_ivalue_map);
  auto inferred_st_sizes = out_tensors[0].sizes().vec();
  // Step2: Create shape tensor and insert to graph inputs.
  auto rand_st_name =
      GetDynamicTensorName(v_rand_shape->debugName(), SHAPE_TENSOR);
  int64_t stack_index = CreateSTAndInsertToDSStack(
      inferred_st_sizes, scalar_indexes, {}, {}, m_dmeta);
  auto rand_st_tensor = graph->addInput(rand_st_name);

  auto symbol_outputshape = c10::Symbol::attr("output_shapes");
  if (aten_rand_node->hasAttribute(symbol_outputshape)) {
    std::string original = aten_rand_node->s(symbol_outputshape);
    std::string modified = original.substr(1, original.length() - 2);
    m_range_infos->emplace_back(
        habana_helpers::RangeInfo({}, {}, modified, "INVALID", -1));
    PT_DYNAMIC_SHAPE_DEBUG("Adding ST ranges shape", modified);
  } else {
    PT_DYNAMIC_SHAPE_WARN(
        "ERROR: Not adding ST ranges output_shapes attribute null");
  }

  // Step3: Create hpu::rand_ds node and insert to the graph
  static const auto hpu_rand_symbol{
      c10::Symbol::fromQualString("hpu::habana_rand_st")};
  CreateAndInsertDynamicNodeToGraph(
      graph,
      aten_rand_node,
      hpu_rand_symbol,
      {aten_rand_node->inputs().at(0),
       rand_st_tensor,
       aten_rand_node->inputs().at(2),
       aten_rand_node->inputs().at(3),
       aten_rand_node->inputs().at(4),
       aten_rand_node->inputs().at(5)},
      value_ivalue_map);

  // Step4: Register patching function and tensor lists
  std::vector<int64_t> dtensor_indexes{stack_index};
  InputPatchPair patch_info(
      &RandnOperatorDS::UpdateDynamicInputs, dtensor_indexes);
  m_dmeta->ds_input_patching_list.push_back(patch_info);
  return true;
}

void RandnOperatorDS::UpdateDynamicInputs(
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
  auto dtensor = dtensor_list[0]->toTensor();
  SymIntData& scalar_idx = scalar_idx_list[0];
  UpdateShapeTensorSize(dtensor, scalar_idx.values, orig_stack, launch_shapes);
}

// Dynamic shape (DS) support for `randint` using shape tensor
bool RandintOperatorDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* aten_rand_node,
    torch::jit::Stack& org_stack,
    GraphInputIndexMap& org_stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  // 8 inputs in randint
  HABANA_ASSERT(8 == aten_rand_node->inputs().size());
  // 1 scalars: size
  auto v_rand_shape = aten_rand_node->inputs().at(3);
  // fetching the list symbol for size
  static const auto list_construct_symbol{
      c10::Symbol::fromQualString("prim::ListConstruct")};
  HABANA_ASSERT(
      v_rand_shape->node()->kind() == list_construct_symbol,
      "randint input is not a ListConstruct, it is: ",
      v_rand_shape->node()->kind().toQualString());
  auto list_construct_node{v_rand_shape->node()};
  auto graph{aten_rand_node->owningGraph()};

  // Step 1: Collect shape and scalar pos used in ListConstruct input node
  std::vector<int64_t> st_size;
  std::vector<int64_t> scalar_indexes;
  GetValuesAndScalarIndexesFromListConstruct(
      list_construct_node,
      org_stack,
      org_stack_index_map,
      st_size,
      scalar_indexes);
  auto out_tensors = getOutputTensers(aten_rand_node, value_ivalue_map);
  auto inferred_st_sizes = out_tensors[0].sizes().vec();
  // Step2: Create shape tensor and insert to graph inputs.
  auto rand_st_name =
      GetDynamicTensorName(v_rand_shape->debugName(), SHAPE_TENSOR);
  int64_t stack_index = CreateSTAndInsertToDSStack(
      inferred_st_sizes, scalar_indexes, {}, {}, m_dmeta);
  auto rand_st_tensor = graph->addInput(rand_st_name);

  auto symbol_outputshape = c10::Symbol::attr("output_shapes");
  if (aten_rand_node->hasAttribute(symbol_outputshape)) {
    std::string original = aten_rand_node->s(symbol_outputshape);
    std::string modified = original.substr(1, original.length() - 2);
    m_range_infos->emplace_back(
        habana_helpers::RangeInfo({}, {}, modified, "INVALID", -1));
    PT_DYNAMIC_SHAPE_DEBUG("Adding ST ranges shape", modified);
  } else {
    PT_DYNAMIC_SHAPE_WARN(
        "ERROR: Not adding ST ranges output_shapes attribute null");
  }

  // Step3: Create hpu::rand_ds node and insert to the graph
  static const auto hpu_rand_symbol{
      c10::Symbol::fromQualString("hpu::habana_randint_st")};
  CreateAndInsertDynamicNodeToGraph(
      graph,
      aten_rand_node,
      hpu_rand_symbol,
      {aten_rand_node->inputs().at(0),
       aten_rand_node->inputs().at(1),
       aten_rand_node->inputs().at(2),
       rand_st_tensor,
       aten_rand_node->inputs().at(4),
       aten_rand_node->inputs().at(5),
       aten_rand_node->inputs().at(6),
       aten_rand_node->inputs().at(7)},
      value_ivalue_map);

  // Step4: Register patching function and tensor lists
  std::vector<int64_t> dtensor_indexes{stack_index};
  InputPatchPair patch_info(
      &RandintOperatorDS::UpdateDynamicInputs, dtensor_indexes);
  m_dmeta->ds_input_patching_list.push_back(patch_info);
  return true;
}

void RandintOperatorDS::UpdateDynamicInputs(
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
  auto dtensor = dtensor_list[0]->toTensor();
  SymIntData& scalar_idx = scalar_idx_list[0];
  UpdateShapeTensorSize(dtensor, scalar_idx.values, orig_stack, launch_shapes);
}

// Dynamic shape (DS) support for `full` using shape tensor
bool FullOpDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* aten_full_node,
    torch::jit::Stack& org_stack,
    GraphInputIndexMap& org_stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  // 6 inputs in full
  HABANA_ASSERT(6 == aten_full_node->inputs().size());
  // 1 scalars: size
  auto v_full_shape = aten_full_node->inputs().at(0);
  // fetching the list symbol for size
  static const auto list_construct_symbol{
      c10::Symbol::fromQualString("prim::ListConstruct")};
  HABANA_ASSERT(
      v_full_shape->node()->kind() == list_construct_symbol,
      "full input is not a ListConstruct, it is: ",
      v_full_shape->node()->kind().toQualString());
  auto list_construct_node{v_full_shape->node()};
  auto graph{aten_full_node->owningGraph()};

  // Step 1: Collect shape and scalar pos used in ListConstruct input node
  std::vector<int64_t> st_size;
  std::vector<int64_t> scalar_indexes;
  GetValuesAndScalarIndexesFromListConstruct(
      list_construct_node,
      org_stack,
      org_stack_index_map,
      st_size,
      scalar_indexes);
  auto out_tensors = getOutputTensers(aten_full_node, value_ivalue_map);
  auto inferred_st_sizes = out_tensors[0].sizes().vec();
  // Step2: Create shape tensor and insert to graph inputs.
  auto full_st_name =
      GetDynamicTensorName(v_full_shape->debugName(), SHAPE_TENSOR);
  int64_t stack_index = CreateSTAndInsertToDSStack(
      inferred_st_sizes, scalar_indexes, {}, {}, m_dmeta);
  auto full_st_tensor = graph->addInput(full_st_name);

  auto symbol_outputshape = c10::Symbol::attr("output_shapes");
  if (aten_full_node->hasAttribute(symbol_outputshape)) {
    std::string original = aten_full_node->s(symbol_outputshape);
    std::string modified = original.substr(1, original.length() - 2);
    m_range_infos->emplace_back(
        habana_helpers::RangeInfo({}, {}, modified, "INVALID", -1));
    PT_DYNAMIC_SHAPE_DEBUG("Adding ST ranges shape", modified);
  } else {
    PT_DYNAMIC_SHAPE_WARN(
        "ERROR: Not adding ST ranges output_shapes attribute null");
  }

  // Step3: Create hpu::full_ds node and insert to the graph
  static const auto hpu_full_symbol{
      c10::Symbol::fromQualString("hpu::full_ds")};
  CreateAndInsertDynamicNodeToGraph(
      graph,
      aten_full_node,
      hpu_full_symbol,
      {full_st_tensor,
       aten_full_node->inputs().at(1),
       aten_full_node->inputs().at(2),
       aten_full_node->inputs().at(3),
       aten_full_node->inputs().at(4),
       aten_full_node->inputs().at(5)},
      value_ivalue_map);

  // Step4: Register patching function and tensor lists
  std::vector<int64_t> dtensor_indexes{stack_index};
  InputPatchPair patch_info(&FullOpDS::UpdateDynamicInputs, dtensor_indexes);
  m_dmeta->ds_input_patching_list.push_back(patch_info);
  return true;
}

void FullOpDS::UpdateDynamicInputs(
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
  auto dtensor = dtensor_list[0]->toTensor();
  SymIntData& scalar_idx = scalar_idx_list[0];
  UpdateShapeTensorSize(dtensor, scalar_idx.values, orig_stack, launch_shapes);
}

// Dynamic shape (DS) support for `empty` using shape tensor
bool EmptyOpDS::ReplaceWithDynamicHPUOp(
    torch::jit::Node* aten_empty_node,
    torch::jit::Stack& org_stack,
    GraphInputIndexMap& org_stack_index_map,
    ValueIvalueMap& value_ivalue_map,
    std::shared_ptr<DynamicGraphMetaData> m_dmeta) {
  // 6 inputs in empty
  HABANA_ASSERT(6 == aten_empty_node->inputs().size());
  // 1 scalars: size
  auto v_empty_shape = aten_empty_node->inputs().at(0);
  // fetching the list symbol for size
  static const auto list_construct_symbol{
      c10::Symbol::fromQualString("prim::ListConstruct")};
  HABANA_ASSERT(
      v_empty_shape->node()->kind() == list_construct_symbol,
      "empty input is not a ListConstruct, it is: ",
      v_empty_shape->node()->kind().toQualString());
  auto list_construct_node{v_empty_shape->node()};
  auto graph{aten_empty_node->owningGraph()};

  // Step 1: Collect shape and scalar pos used in ListConstruct input node
  std::vector<int64_t> st_size;
  std::vector<int64_t> scalar_indexes;
  GetValuesAndScalarIndexesFromListConstruct(
      list_construct_node,
      org_stack,
      org_stack_index_map,
      st_size,
      scalar_indexes);
  auto out_tensors = getOutputTensers(aten_empty_node, value_ivalue_map);
  auto inferred_st_sizes = out_tensors[0].sizes().vec();
  // Step2: Create shape tensor and insert to graph inputs.
  auto empty_st_name =
      GetDynamicTensorName(v_empty_shape->debugName(), SHAPE_TENSOR);
  int64_t stack_index = CreateSTAndInsertToDSStack(
      inferred_st_sizes, scalar_indexes, {}, {}, m_dmeta);
  auto empty_st_tensor = graph->addInput(empty_st_name);
  auto symbol_outputshape = c10::Symbol::attr("output_shapes");
  if (aten_empty_node->hasAttribute(symbol_outputshape)) {
    std::string original = aten_empty_node->s(symbol_outputshape);
    std::string modified = original.substr(1, original.length() - 2);
    m_range_infos->emplace_back(
        habana_helpers::RangeInfo({}, {}, modified, "INVALID", -1));
    PT_DYNAMIC_SHAPE_DEBUG("Adding ST ranges shape", modified);
  } else {
    PT_DYNAMIC_SHAPE_WARN(
        "ERROR: Not adding ST ranges output_shapes attribute null");
  }

  // Step3: Create hpu::empty_ds node and insert to the graph
  static const auto hpu_empty_symbol{
      c10::Symbol::fromQualString("hpu::empty_ds")};
  CreateAndInsertDynamicNodeToGraph(
      graph,
      aten_empty_node,
      hpu_empty_symbol,
      {empty_st_tensor,
       aten_empty_node->inputs().at(1),
       aten_empty_node->inputs().at(2),
       aten_empty_node->inputs().at(3),
       aten_empty_node->inputs().at(4),
       aten_empty_node->inputs().at(5)},
      value_ivalue_map);

  // Step4: Register patching function and tensor lists
  std::vector<int64_t> dtensor_indexes{stack_index};
  InputPatchPair patch_info(&EmptyOpDS::UpdateDynamicInputs, dtensor_indexes);
  m_dmeta->ds_input_patching_list.push_back(patch_info);
  return true;
}

void EmptyOpDS::UpdateDynamicInputs(
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
  auto dtensor = dtensor_list[0]->toTensor();
  SymIntData& scalar_idx = scalar_idx_list[0];
  UpdateShapeTensorSize(dtensor, scalar_idx.values, orig_stack, launch_shapes);
}

} // namespace graph
} // namespace habana
