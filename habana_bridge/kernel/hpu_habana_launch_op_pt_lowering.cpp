/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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

#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"

#include "pytorch_helpers/habana_device//hpu_cached_devices.h"
#include "pytorch_helpers/habana_helpers/logging.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

void habana::HabanaLaunchOpPT::CopyInputStack(torch::jit::Stack& input_st) {
  // Keep a handle to the stack for future use
  pt_stack = &input_st;

  for (size_t i = 0; i < pt_stack->size(); i++) {
    if (pt_stack->at(i).isTensor()) {
      auto& t{pt_stack->at(i).toTensor()};
      input_tms.emplace_back(
          t.sizes().vec(), t.strides().vec(), t.suggest_memory_format());
    } else {
      input_tms.emplace_back(
          std::vector<int64_t>(),
          std::vector<int64_t>(),
          c10::MemoryFormat::Preserve);
    }

    IValPtrShared ivpsh = std::make_shared<IVal>(pt_stack->at(i));
    pt_stack_sh.push_back(ivpsh);
    if (ivpsh->isTensor() || ivpsh->isTensorList()) {
      num_tensor_inputs++;
    }
  }
}

void habana::HabanaLaunchOpPT::Clear(bool is_shape_inference) {
  if (is_shape_inference == false) {
    pt_stack = nullptr;
    pt_stack_sh.clear();
    num_tensor_inputs = 0;
    habana::ShapeInference::Reset();
  }

  value_to_ivalue.clear();
  watchlist_.clear();
  syn_graph_ptr = nullptr;
  cur_rvalpsh = nullptr;

  habana_kernels.clear();

  input_tivs.clear();
  duplicate_input_tivs.clear();
  input_tiv_map.clear();

  intermediate_tinfos.clear();
  dma_input_tensorinfos.clear();
  shape_tensor_tinfos.clear();
  output_tensorinfos.clear();
  duplicate_outtinfos.clear();
  duplicate_input_to_outtinfo_map.clear();
  duplicate_intermediate_to_outtinfo_map.clear();

  aten_intermediates.clear();
  aten_dma_inputs.clear();

  output_tensorinfo_map.clear();

  valptr_to_persistent_map.clear();

  pt_to_synapse_tensors.clear();
  meta_syn_tensors.clear();
  buff_to_input_ivpsh_map.clear();
  buff_to_intermediate_ivpsh_map.clear();
  buff_to_output_ivpsh_map.clear();
  buff_to_syn_tensor_map.clear();

  jit_to_synapse_node_idx_map.clear();
}

void habana::HabanaLaunchOpPT::CompileSynapseGraph() {
  // Process control edges
  HabanaLaunchOpPT::ProcessControlEdges();

  TORCH_CHECK(syn_graph_ptr, "Synapse graph pointer is null");
  if (syn_graph_ptr->is_empty()) {
    PT_BRIDGE_DEBUG("Empty synapse graph. Nothing to compile.");
    return;
  }

  std::chrono::steady_clock::time_point t_start;
  t_start = std::chrono::steady_clock::now();
  auto&& error_variant{syn_graph_ptr->compile()};
  auto t_compile = std::chrono::steady_clock::now() - t_start;
  t_compile_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_compile).count();

  if (ABSL_PREDICT_FALSE(
          absl::holds_alternative<synapse_helpers::synapse_error>(
              error_variant))) {
    auto& error = absl::get<synapse_helpers::synapse_error>(error_variant);
    PT_BRIDGE_FATAL(
        "syn compile encountered : ",
        error.error,
        " ",
        error.status,
        " compile time ",
        t_compile_ns,
        " ns");
  }

  RecipeValueSpec::increment_compile_count();

  auto cur_recipe = get_value(std::move(error_variant));
  cur_rvalpsh = std::make_shared<RecipeValueSpec>(cur_recipe, jit_ir_graph);
  RecipeValueSpec& rv = *cur_rvalpsh;

  // Get workspace size of the compiled recipe
  auto&& ws_size_result{
      synapse_helpers::graph::query_workspace_size(*cur_recipe)};
  if (ABSL_PREDICT_FALSE(
          absl::holds_alternative<synapse_helpers::synapse_error>(
              ws_size_result))) {
    auto& error = absl::get<synapse_helpers::synapse_error>(ws_size_result);
    PT_BRIDGE_FATAL(
        "workspace size query failed: ", error.error, " ", error.status);
    TORCH_CHECK(false, "workspace size query failed");
  }
  rv.workspace_size = get_value(ws_size_result);
}

void habana::HabanaLaunchOpPT::ConstructPatchingTable() {
  TORCH_CHECK(syn_graph_ptr, "Synapse graph pointer is null");
  if (syn_graph_ptr->is_empty()) {
    PT_BRIDGE_DEBUG(
        "Empty synapse graph. No need to construct the patching table.");
    return;
  }

  TORCH_CHECK(cur_rvalpsh, "Recipe pointer is null");
  RecipeValueSpec& rv = *cur_rvalpsh;

  // input_tivs need to be reordered for patching
  OrderInputs();

  // rv.num_inputs and rv.num_induplicates will be set by
  // FlattenAndLinkInputTIVs
  FlattenAndLinkInputTIVs(rv);

  if (!dma_input_tensorinfos.empty()) {
    rv.num_dma_inputs = dma_input_tensorinfos.size();
    rv.dtensorinfos->insert(
        rv.dtensorinfos->end(),
        dma_input_tensorinfos.begin(),
        dma_input_tensorinfos.end());
  }

  if (!shape_tensor_tinfos.empty()) {
    rv.num_shape_tensors = shape_tensor_tinfos.size();
    rv.dtensorinfos->insert(
        rv.dtensorinfos->end(),
        shape_tensor_tinfos.begin(),
        shape_tensor_tinfos.end());
  }

  // tinfos for outputs are populated during compile
  // need to be reordered only when the tensor handles are released
  rv.aten_outputs = std::make_shared<std::vector<IValPtrShared>>(
      std::vector<IValPtrShared>());
  if (!enable_caching_) {
    if (!intermediate_tinfos.empty()) {
      rv.num_intermediates = intermediate_tinfos.size();
      rv.dtensorinfos->insert(
          rv.dtensorinfos->end(),
          intermediate_tinfos.begin(),
          intermediate_tinfos.end());
    }

    // At this point, tinfos for inputs, input duplicates and intermediates
    // are populated
    TORCH_CHECK(
        (rv.num_inputs + rv.num_induplicates + rv.num_dma_inputs +
             rv.num_shape_tensors + rv.num_intermediates ==
         rv.dtensorinfos->size()),
        "num_inputs ",
        rv.num_inputs,
        " num_induplicates ",
        rv.num_induplicates,
        " num_dma_inputs ",
        rv.num_dma_inputs,
        " num_intermediates ",
        rv.num_intermediates,
        " num_shape_tensors ",
        rv.num_shape_tensors,
        " are not adding up to #dtensorinfos ",
        rv.dtensorinfos->size());

    rv.dtensorinfos->insert(
        rv.dtensorinfos->end(),
        output_tensorinfos.begin(),
        output_tensorinfos.end());

    rv.num_outputs = output_tensorinfos.size();
    rv.num_tinfos = rv.dtensorinfos->size();

    for (auto output : jit_ir_graph->outputs()) {
      auto oit = value_to_ivalue.find(output);
      TORCH_CHECK(
          oit != value_to_ivalue.end(),
          "value_to_ivalue does not have an entry for %",
          output->debugName());
      IValPtrShared ivpsh = oit->second;
      rv.aten_outputs->push_back(ivpsh);
    }
  } else {
    // TODO :
    //   preclude any interim tinfo from adding to output_tensorinfo_map
    OrderOutputTinfos(rv);
  }

  for (auto& ti : *rv.dtensorinfos) {
    if (!ti.is_duplicate()) {
      rv.ntensorbytes += ti.get_size();
    }
  }

  // At this point, tinfos for inputs, input duplicates, dma_inputs,
  // intermediates, outputs and output duplicates are populated
  auto total_tinfos = rv.num_inputs + rv.num_induplicates + rv.num_dma_inputs +
      rv.num_shape_tensors + rv.num_intermediates + rv.num_outputs +
      rv.num_outduplicates + rv.num_input_to_outduplicates +
      rv.num_intermediate_to_outduplicates + rv.num_output_to_outduplicates;

  TORCH_CHECK(
      total_tinfos == rv.dtensorinfos->size(),
      __LINE__,
      " ::",
      " num_inputs ",
      rv.num_inputs,
      " num_induplicates ",
      rv.num_induplicates,
      " num_dma_inputs ",
      rv.num_dma_inputs,
      " num_intermediates ",
      rv.num_intermediates,
      " num_outputs ",
      rv.num_outputs,
      " num_outduplicates ",
      rv.num_outduplicates,
      " num_input_to_outduplicates ",
      rv.num_input_to_outduplicates,
      " num_intermediate_to_outduplicates ",
      rv.num_intermediate_to_outduplicates,
      " num_output_to_outduplicates ",
      rv.num_output_to_outduplicates,
      " are not adding up to #dtensorinfos ",
      rv.dtensorinfos->size());

  rv.populate_syn_tensor_ids();
  rv.key = cur_rargpsh->hashCode();
}

void habana::HabanaLaunchOpPT::DumpTensors_pre(RecipeValueSpec& rv) {
  if (enable_tensor_dump_) {
    std::ofstream tensor_file;
    tensor_file.open(
        tdmp_file_name_pre_.c_str(), std::ios::out | std::ios::app);
    for (size_t i = 0; i < rv.num_tinfos; ++i) {
      if (rv.dtensorinfos->at(i).watch_enabled()) {
        rv.d2h_dbuff(i);
        rv.print_hbuff(i, tensor_file, iteration_count_, tensor_dump_numel_);
      }
    }
    tensor_file.close();
  }
}

void habana::HabanaLaunchOpPT::DumpTensors(RecipeValueSpec& rv) {
  if (enable_tensor_dump_) {
    std::ofstream tensor_file;
    tensor_file.open(tdmp_file_name_.c_str(), std::ios::out | std::ios::app);
    for (size_t i = 0; i < rv.num_tinfos; ++i) {
      if (rv.dtensorinfos->at(i).watch_enabled()) {
        rv.d2h_dbuff(i);
        rv.print_hbuff(i, tensor_file, iteration_count_, tensor_dump_numel_);
      }
    }
    tensor_file.close();
  }
}

void habana::HabanaLaunchOpPT::ExecuteSynapseGraph() {
  TORCH_CHECK(syn_graph_ptr, "Synapse graph pointer is null");
  if (syn_graph_ptr->is_empty()) {
    PT_BRIDGE_DEBUG("Empty synapse graph. Will update outputs directly.");
    UpdateOutputs();
    return;
  }

  auto& device = synapse_helpers::HPURegistrar::get_device();
  synDeviceId device_id = device.id();

  TORCH_CHECK(cur_rvalpsh, "Recipe pointer is null");
  RecipeValueSpec& rv = *cur_rvalpsh;

  if (enable_tensor_dump_) {
    if (0 == htensor_wbuff_size) {
      for (size_t i = 0; i < rv.num_tinfos; ++i) {
        htensor_wbuff_size =
            std::max(htensor_wbuff_size, rv.dtensorinfos->at(i).get_size());
      }
    }

    if (!htensor_wbuff) {
      synStatus status;
      status = synHostMalloc(
          device_id, htensor_wbuff_size, 0, (void**)&(htensor_wbuff));
      TORCH_CHECK(status == synSuccess, "host-malloc failed");
    }
    rv.htensor_wbuff = htensor_wbuff;
    rv.htensor_wbuff_size = htensor_wbuff_size;
  }

  if (enable_tensor_dump_) {
    DumpTensors_pre(rv);
  }

  if (refine_ds_enabled_) {
    // Initiate recipe execution time collection
    InitiateSynlaunchTimeCapture(rv);
    // Add the jit_ir_graph to current_dbipsh_
    current_dbipsh_->SetJitIRGraphPtr(jit_ir_graph);
    current_dbipsh_->UpdateCompileTime(t_compile_ns, current_bucket_id_);
  }

  PT_BRIDGE_DEBUG(
      "HabanaOp recipe cache :: launching new recipe", rv.header_str());

  std::shared_ptr<std::vector<IValPtrShared>> intermediate_tensors_ptr =
      std::make_shared<std::vector<IValPtrShared>>(
          std::vector<IValPtrShared>());

  for (auto& tensor : aten_intermediates) {
    IValPtrShared ivpsh = std::make_shared<IVal>(tensor);
    intermediate_tensors_ptr->push_back(ivpsh);
  }

  rv.launch(input_refs, intermediate_tensors_ptr);
  rv.update_hit_count();

  if (enable_tensor_dump_) {
    DumpTensors(rv);
  }

  if (enable_caching_) {
    // Add the <key,value> pair to the map
    if (refine_ds_enabled_) {
      rv.dynamic_graph = syn_graph_ptr->is_dynamic_graph();
      // Add the recipe to the corresponding bucket
      current_dbipsh_->SetSynapseRecipePtr(current_bucket_id_, cur_rvalpsh);
    }
    RecipeCacheLRU::get_cache().add(cur_rargpsh, cur_rvalpsh);
    PT_BRIDGE_DEBUG(
        "HabanaOp recipe cache :: adding new recipe to cache :: ", rv.key);
  }
  if (enable_caching_ && refine_ds_enabled_) {
    PT_DYNAMIC_SHAPE_DEBUG(
        current_dbipsh_->digest_str(),
        "Recipe Header::",
        rv.header_str(),
        "\n",
        rv.digest_str(),
        "\n",
        "--------------------");
  } else {
    PT_BRIDGE_DEBUG(rv.digest_str());
  }

  UpdateOutputs(rv);
}

void habana::HabanaLaunchOpPT::FlattenAndLinkInputTIVs(RecipeValueSpec& rv) {
  // dtensorinfos maintain the flattened tinfo list
  rv.dtensorinfos =
      std::make_shared<std::vector<PtTensorInfo>>(std::vector<PtTensorInfo>());

  std::unordered_map<void*, size_t> buff_to_inputtividx_map;
  for (auto& tiv : input_tivs) {
    if (absl::holds_alternative<PtTensorInfo>(tiv)) {
      const auto ti = absl::get<PtTensorInfo>(tiv);
      rv.dtensorinfos->push_back(ti);
      if (enable_caching_) {
        void* buffp = ti.get_buffer_start();
        buff_to_inputtividx_map.emplace(buffp, rv.dtensorinfos->size() - 1);
      }
    } else if (absl::holds_alternative<std::vector<PtTensorInfo>>(tiv)) {
      for (const auto& ti : absl::get<std::vector<PtTensorInfo>>(tiv)) {
        rv.dtensorinfos->push_back(ti);
        if (enable_caching_) {
          void* buffp = ti.get_buffer_start();
          buff_to_inputtividx_map.emplace(buffp, rv.dtensorinfos->size() - 1);
        }
      }
    } else {
      TORCH_CHECK(false, "Error condition for input tiv");
    }
  }
  // At this point inputs tinfos are populated
  rv.num_inputs = rv.dtensorinfos->size();

  // Link the input tivs with the duplicate
  size_t nduplicates{0};
  for (auto& tiv : duplicate_input_tivs) {
    if (absl::holds_alternative<PtTensorInfo>(tiv)) {
      auto ti = absl::get<PtTensorInfo>(tiv);
      if (enable_caching_) {
        void* buffp = ti.get_buffer_start();
        auto it_parent = buff_to_inputtividx_map.find(buffp);

        std::ostringstream err;
        err << ti;

        TORCH_CHECK(
            buff_to_inputtividx_map.end() != it_parent,
            "parent tinfo is missing for input duplicate ",
            err.str());

        ti.set_duplicate_flag(true);
        size_t parent_idx = it_parent->second;
        TORCH_CHECK(
            parent_idx < num_inputs,
            "out of bound parent index : ",
            parent_idx,
            " for ",
            ti.get_syn_name());
        ti.set_parent_index(parent_idx);
        PT_BRIDGE_DEBUG(
            "FlattenAndLinkInputTIVs: Input duplicate: parent idx ",
            parent_idx,
            " parent buffer ptr ",
            rv.dtensorinfos->at(parent_idx).get_buffer(),
            " duplicate_tiv buffer ptr ",
            ti.get_buffer());
      }
      rv.dtensorinfos->push_back(ti);
      nduplicates++;
    } else {
      TORCH_CHECK(false, "duplicate tiv must be a tensor");
    }
  }
  TORCH_CHECK(
      nduplicates == duplicate_input_tivs.size(),
      "#duplicate_input_tivs ",
      duplicate_input_tivs.size(),
      " is not matching with num_induplicates ",
      nduplicates);

  rv.num_induplicates = nduplicates;

  // At this point inputs and duplicate tinfos are populated
  TORCH_CHECK(
      (rv.num_inputs + rv.num_induplicates == rv.dtensorinfos->size()),
      "num_inputs ",
      rv.num_inputs,
      "num_induplicates ",
      rv.num_induplicates,
      " are not adding up to #dtensorinfos ",
      rv.dtensorinfos->size());
}

void habana::HabanaLaunchOpPT::OrderInputs() {
  if (enable_caching_) {
    // Order the input_tivs according to the order of suggraph inputs
    size_t i = pt_stack_sh.size() - num_inputs;
    for (; i < pt_stack_sh.size(); i++) {
      IValPtrShared ivpsh = pt_stack_sh.at(i);
      if (ivpsh->isTensor() || ivpsh->isTensorList()) {
        auto it = input_tiv_map.find(ivpsh);
        if (it != input_tiv_map.end()) {
          input_tivs.push_back(it->second);
        } else {
          TORCH_CHECK(false, "synapse tensor not found for input index", i);
        }
      }
    }
    TORCH_CHECK(
        input_tivs.size() == num_tensor_inputs,
        "number of input tensors ",
        num_tensor_inputs,
        " mismatch with #input_tivs ",
        input_tivs.size());
  }
}

void habana::HabanaLaunchOpPT::OrderOutputTinfos(RecipeValueSpec& rv) {
  bool has_empty_name = false;

  std::unordered_map<void*, size_t> buff_to_outputtinfoidx_map;
  // push the actual output tinfos
  size_t output_idx{0};
  for (auto output : jit_ir_graph->outputs()) {
    auto oit = value_to_ivalue.find(output);
    TORCH_CHECK(
        oit != value_to_ivalue.end(),
        "value_to_ivalue does not have an entry for %",
        output->debugName());

    IValPtrShared ivpsh = oit->second;
    TORCH_CHECK(nullptr != ivpsh, "IValPtrShared for subgraph output is null");

    // Checking where we can find the outputs
    {
      if (output_tensorinfo_map.count(ivpsh)) {
        auto it = output_tensorinfo_map.find(ivpsh);
        it->second.set_output_index(output_idx);
        output_tensorinfos.push_back(it->second);
        if (it->second.get_syn_name().empty()) {
          has_empty_name = true;
        }
        output_tensorinfo_map.erase(ivpsh);
      } else if (duplicate_input_to_outtinfo_map.count(ivpsh)) {
        auto it_dup = duplicate_input_to_outtinfo_map.find(ivpsh);
        it_dup->second.set_output_index(output_idx);
      } else if (duplicate_intermediate_to_outtinfo_map.count(ivpsh)) {
        auto it_dup = duplicate_intermediate_to_outtinfo_map.find(ivpsh);
        it_dup->second.set_output_index(output_idx);
      } else if (duplicate_output_to_outtinfo_map.count(ivpsh)) {
        auto it_dup = duplicate_output_to_outtinfo_map.find(ivpsh);
        it_dup->second.set_output_index(output_idx);
      } else {
        TORCH_CHECK(
            0,
            "Unaccounted output %",
            output->debugName(),
            " at index ",
            output_idx,
            ". Cached recipe execution might break");
      }
    }

    // add ivpsh to outputs
    rv.aten_outputs->push_back(ivpsh);
    output_idx++;
  }

  TORCH_CHECK(!has_empty_name, "empty tensor name");

  size_t intermediates_start = rv.num_inputs + rv.num_induplicates +
      rv.num_dma_inputs + rv.num_shape_tensors;

  TORCH_CHECK(
      output_tensorinfo_map.empty(),
      "output_tensorinfo_map still contains ",
      output_tensorinfo_map.size(),
      " tensors.");

  // Add the intermediates to rv.dtensorinfos
  std::unordered_map<void*, size_t> buff_to_interim_tividx_map;
  size_t interim_tinfo_idx{intermediates_start};
  if (!intermediate_tinfos.empty()) {
    rv.num_intermediates = intermediate_tinfos.size();
    for (auto& ti : intermediate_tinfos) {
      void* buffp = ti.get_buffer_start();
      // Duplicate analysis for the persistent intermediates
      if (buff_to_interim_tividx_map.count(buffp)) {
        ti.set_duplicate_flag(true);
        ti.set_parent_index(buff_to_interim_tividx_map[buffp]);
      } else {
        buff_to_interim_tividx_map.emplace(buffp, interim_tinfo_idx);
      }
      interim_tinfo_idx++;
    }

    rv.dtensorinfos->insert(
        rv.dtensorinfos->end(),
        intermediate_tinfos.begin(),
        intermediate_tinfos.end());
  }

  // At this point, within dtensorinfos, tinfos for inputs,
  // input duplicates and intermediates are added.
  size_t intermediates_end = intermediates_start + rv.num_intermediates;
  size_t outputs_start = intermediates_end;

  // Add the outputs to rv.dtensorinfos
  for (auto& ti : output_tensorinfos) {
    rv.dtensorinfos->push_back(ti);
    void* buffp = ti.get_buffer_start();

    buff_to_outputtinfoidx_map.emplace(buffp, rv.dtensorinfos->size() - 1);
  }
  rv.num_outputs = output_tensorinfos.size();

  // At this point, within dtensorinfos, tinfos for inputs, input_duplicates,
  // intermediates and outputs are added.
  size_t outputs_end = outputs_start + rv.num_outputs;

  // Link the out tivs with the duplicate.
  // Remember these are not outputs but tensors
  // that go back to the FusedOp from the outputs
  size_t nduplicates{0};
  for (auto& ti : duplicate_outtinfos) {
    void* buffp = ti.get_buffer_start();
    auto it_parent = buff_to_outputtinfoidx_map.find(buffp);

    std::ostringstream err;
    err << ti;

    TORCH_CHECK(
        buff_to_outputtinfoidx_map.end() != it_parent,
        "parent tinfo is missing for output duplicate ",
        err.str());

    ti.set_duplicate_flag(true);
    size_t parent_idx = it_parent->second;
    TORCH_CHECK(
        parent_idx >= outputs_start && parent_idx < outputs_end,
        "for output duplicate ",
        ti.get_syn_name(),
        "parent index should be within [",
        outputs_start,
        ',',
        outputs_end,
        ')');
    ti.set_parent_index(parent_idx);
    rv.dtensorinfos->push_back(ti);
    nduplicates++;
  }
  rv.num_outduplicates = nduplicates;

  // Create the input tensor tiv to idx map, this is required
  // to match the input to output duplicates against their parent idx.
  std::unordered_map<void*, size_t> buff_to_inputtividx_map;
  size_t in_idx = 0;
  for (auto& tiv : input_tivs) {
    if (absl::holds_alternative<PtTensorInfo>(tiv)) {
      const auto ti = absl::get<PtTensorInfo>(tiv);
      void* buffp = ti.get_buffer_start();
      buff_to_inputtividx_map.emplace(buffp, in_idx++);
    } else if (absl::holds_alternative<std::vector<PtTensorInfo>>(tiv)) {
      for (const auto& ti : absl::get<std::vector<PtTensorInfo>>(tiv)) {
        void* buffp = ti.get_buffer_start();
        buff_to_inputtividx_map.emplace(buffp, in_idx++);
      }
    } else {
      TORCH_CHECK(false, "Error condition for input tiv");
    }
  }

  // Link the inout tivs with the duplicate
  // These are tensors that are duplicated from an input and is part
  // of the graph output
  nduplicates = 0;
  for (auto& mi : duplicate_input_to_outtinfo_map) {
    auto ivpsh = mi.first;
    auto& ti = mi.second;
    void* buffp = ti.get_buffer_start();
    auto it_parent = buff_to_inputtividx_map.find(buffp);

    std::ostringstream err;
    err << ti;

    TORCH_CHECK(
        buff_to_inputtividx_map.end() != it_parent,
        "parent tinfo is missing for input_to_out duplicate ",
        err.str());

    ti.set_duplicate_flag(true);
    auto parent_idx = it_parent->second;
    TORCH_CHECK(
        parent_idx < rv.num_inputs,
        "for in_to_out duplicate ",
        ti.get_syn_name(),
        "parent index ",
        parent_idx,
        " should be within [",
        0,
        ',',
        rv.num_inputs,
        ')');
    ti.set_parent_index(parent_idx);
    rv.dtensorinfos->push_back(ti);
    nduplicates++;
  }

  rv.num_input_to_outduplicates = nduplicates;

  // Link the out tivs which are duplicate of persistent intermediates
  // These are tensors that are duplicated from a persistent interim and is
  // part of the graph output
  nduplicates = 0;
  for (auto& mi : duplicate_intermediate_to_outtinfo_map) {
    auto ivpsh = mi.first;
    auto& ti = mi.second;
    void* buffp = ti.get_buffer_start();
    auto it_parent = buff_to_interim_tividx_map.find(buffp);

    std::ostringstream err;
    err << ti;

    TORCH_CHECK(
        buff_to_interim_tividx_map.end() != it_parent,
        "parent tinfo is missing for interim_to_out duplicate ",
        err.str());

    ti.set_duplicate_flag(true);
    auto parent_idx = it_parent->second;
    TORCH_CHECK(
        (parent_idx >= intermediates_start && parent_idx < intermediates_end),
        "for interim to out duplicate ",
        ti.get_syn_name(),
        "parent index ",
        parent_idx,
        " should be within [",
        intermediates_start,
        ',',
        intermediates_end,
        ')');
    ti.set_parent_index(parent_idx);
    rv.dtensorinfos->push_back(ti);
    nduplicates++;
  }
  rv.num_intermediate_to_outduplicates = nduplicates;

  // Link the out tivs which are duplicate of actual outputs
  // These are tensors that are duplicated from outputs created by individual
  // ops using aten::empty like calls and is part of the graph output
  nduplicates = 0;
  for (auto& mi : duplicate_output_to_outtinfo_map) {
    auto ivpsh = mi.first;
    auto& ti = mi.second;
    void* buffp = ti.get_buffer_start();
    auto it_parent = buff_to_outputtinfoidx_map.find(buffp);

    std::ostringstream err;
    err << ti;

    TORCH_CHECK(
        buff_to_outputtinfoidx_map.end() != it_parent,
        "parent tinfo is missing for output_to_out duplicate ",
        err.str());

    ti.set_duplicate_flag(true);
    auto parent_idx = it_parent->second;
    TORCH_CHECK(
        parent_idx >= outputs_start && parent_idx < outputs_end,
        parent_idx < rv.num_inputs,
        "for out_to_out duplicate ",
        ti.get_syn_name(),
        "parent index ",
        parent_idx,
        " should be within [",
        outputs_start,
        ',',
        outputs_end,
        ')');
    ti.set_parent_index(parent_idx);
    rv.dtensorinfos->push_back(ti);
    nduplicates++;
  }
  rv.num_output_to_outduplicates = nduplicates;

  rv.num_tinfos = rv.dtensorinfos->size();
}

void habana::HabanaLaunchOpPT::RestoreInputTensorMetadata() {
  for (size_t i{0}; i < pt_stack->size(); i++) {
    if (pt_stack->at(i).isTensor()) {
      auto& input_tensor = pt_stack->at(i).toTensor();
      input_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
          input_tms.at(i).sizes, input_tms.at(i).strides);
      input_tensor.unsafeGetTensorImpl()->empty_tensor_restride(
          input_tms.at(i).mf);
    }
  }
}

void habana::HabanaLaunchOpPT::UpdateOutputs() {
  // Restore the metadata of the inputs
  RestoreInputTensorMetadata();

  // Update the stack from the JIT IR outputs
  torch::jit::drop(*pt_stack, num_inputs);
  for (auto output : jit_ir_graph->outputs()) {
    auto oit = value_to_ivalue.find(output);
    TORCH_CHECK(
        oit != value_to_ivalue.end(),
        "value_to_ivalue does not have an entry for %",
        output->debugName());
    IValPtrShared ivpsh = oit->second;
    pt_stack->insert(pt_stack->end(), *ivpsh);
  }
}

void habana::HabanaLaunchOpPT::UpdateOutputs(RecipeValueSpec& rv) {
  // Restore the metadata of the inputs
  RestoreInputTensorMetadata();

  // Update the stack from the recipe itself
  torch::jit::drop(*pt_stack, num_inputs);
  for (const auto& ivpsh : *(rv.aten_outputs)) {
    pt_stack->insert(pt_stack->end(), *ivpsh);
  }
  rv.aten_outputs = nullptr;
}

void habana::HabanaLaunchOpPT::ProcessInputStack(torch::jit::Stack& input_st) {
  num_inputs = jit_ir_graph->inputs().size();
  TORCH_CHECK(
      num_inputs == input_st.size(),
      "Input stack size=",
      num_inputs,
      " is not matching with #graph_inputs=",
      input_st.size());

  num_tensor_inputs = 0;
  input_refs = torch::jit::last(input_st, num_inputs);

  // All tensors should be on Habana, we should assert otherwise
  bool is_all_hpu = true;
  for (auto& input : input_refs) {
    if (input.isTensor()) {
      is_all_hpu = input.toTensor().device().type() != c10::DeviceType::HPU
          ? false
          : is_all_hpu;
    }
  }

  // We dont support running some ops on CPU while running fused op on Habana
  // All tensors should be alocated to habana before entering this phase
  TORCH_CHECK(
      is_all_hpu == true, " Habana Fusion needs all tensors to be in HPU ");

  // Set the habana operators to capture data
  if (refine_ds_enabled_) {
    habana::ShapeInference::Capture(&m_map_shape);
  }

  CopyInputStack(input_st);
}
