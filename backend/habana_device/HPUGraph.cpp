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
#include "HPUGraph.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/lazy_executor.h"
#include "habana_lazy/view_utils.h"

namespace at {
namespace hpu {

template <typename T>
inline bool isExists(
    const std::unordered_set<T>& setContainer,
    const T& element) {
  return (setContainer.count(element) > 0);
}

HPUGraph::HPUGraph()
    // HPUStreams may not be default-constructed.
    : capture_stream_(c10::hpu::getCurrentHPUStream()) {}

void HPUGraph::capture_begin(bool dry_run) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (capturing_) {
    // already captured started, error. only one graph
    // capture is suported.
    PT_DEVICE_FATAL("GRAPH:: graph Capture already in progress");
    return;
  }
  auto stream = c10::hpu::getCurrentHPUStream();
  habana_lazy::HbExecutionContext* context =
      habana_lazy::get_device_lazy_execution_context();
  capture_stream_ = stream;
  /*flush current Accumulated graph, before capture */
  PT_IRGRAPH_DEBUG("step marker due to new HPUGraph::capture_begin");
  habana_lazy::HbLazyTensor::StepMarker({});

  dynamic_env_ = habana_helpers::GetRefineDynamicShapeStatus();
  if (dynamic_env_) {
    habana_helpers::DisableRefineDynamicShape();
  }

  capturing_ = true;

  /* Set graph capture mode on */
  context->setCapturing(true);
  /* pass this struct to global context */
  context->setCaptureGraph(this);
  context->setDryRun(dry_run);
}

void HPUGraph::capture_end() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (capturing_ == false) {
    // need to start the capture.
    PT_DEVICE_DEBUG("GRAPH:: Use Graph capture to Begin the capture");
    return;
  }
  auto stream = c10::hpu::getCurrentHPUStream();

  if (stream != capture_stream_) {
    PT_DEVICE_FATAL("GRAPH:: Capture must end on the same stream it began on.");
    return;
  }

  habana_lazy::HbExecutionContext* context =
      habana_lazy::get_device_lazy_execution_context();

  /*flush graph to capture in the end */
  PT_IRGRAPH_DEBUG("step marker due to HPUGraph::capture_end");
  habana_lazy::HbLazyTensor::StepMarker({});
  capturing_ = false;

  /* Set graph capture mode off */
  context->setCapturing(false);
  context->setCaptureGraph(nullptr);
  context->setDryRun(false);

  // Save all input lazy tensors and free the output IR values
  for (size_t i = 0; i < captured_graphs.size(); i++) {
    auto single_graph = captured_graphs[i];
    auto num_inputs = single_graph->input_vals_.size();
    single_graph->hblazy_tensors_in_.clear();
    for (size_t inp = 0; inp < num_inputs; ++inp) {
        std::shared_ptr<habana_lazy::Data> d =
            single_graph->input_vals_[inp].m_data_ptr.lock();
        single_graph->hblazy_tensors_in_.emplace_back(
            habana_lazy::HbLazyTensor(std::move(d)));
      }
    }

  // Clear the user marked inputs list
  context->ClearHPUGraphUserMarkedInputs();

  // Not enabling DS back once HPU graph detected
  /*if (dynamic_env_) {
    habana_helpers::EnableRefineDynamicShape();
  }*/
}

void HPUGraph::destroy() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  // Clear all captured SingleHpuGraphs
  captured_graphs.clear();

  habana_lazy::HbExecutionContext* context =
      habana_lazy::get_device_lazy_execution_context();
  if (context != nullptr) {
    /* Set graph capture mode off */
    context->setCapturing(false);
    context->setCaptureGraph(nullptr);
  }
}

void HPUGraph::mark_step() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (capturing_ == false) {
    // need to start the capture.
    PT_DEVICE_DEBUG("GRAPH:: Use Graph capture to Begin the capture");
    return;
  }
  auto stream = c10::hpu::getCurrentHPUStream();

  if (stream != capture_stream_) {
    PT_DEVICE_FATAL("GRAPH:: Capture must end on the same stream it began on.");
    return;
  }

  habana_lazy::HbExecutionContext* context =
      habana_lazy::get_device_lazy_execution_context();

  context->JoinPendingLaunchThread();
  if (context->getGraph()) {
    auto captured_graph = std::make_shared<SingleHPUGraph>(
        context->getRecipeArgSpec(),
        context->getGraph(),
        context->getInputs(),
        context->getOutputs(),
        context->getHbLazyTensors(),
        context->getUserInputIndices(),
        context->getSeedTensorMap(),
        context->getHash(),
        context->getGraphKey(),
        context->getOpStrs());
    captured_graphs.push_back(captured_graph);
    auto user_inp_match = context->getUserInputMatchIndices();
    user_input_match_indices_.insert(
        user_inp_match.begin(), user_inp_match.end());
    PT_IRGRAPH_DEBUG("GRAPH:: captured graph");
    PT_HPUGRAPH_DEBUG(
        "GRAPH:: captured input size ", captured_graph->input_vals_.size());
    PT_HPUGRAPH_DEBUG(
        "GRAPH:: captured output size ", captured_graph->output_vals_.size());
    PT_HPUGRAPH_DEBUG(
        "GRAPH:: captured hblazy_tensors_out_ size ",
        captured_graph->hblazy_tensors_out_.size());
  }
  context->getSeedTensorMap().clear();
  context->resetGraph();
}

/* Find all the lazy tensor id of inputs
 it should not be input of any subsequent graph
 for example graph 1 - input x, inplace op on x
 graph 2- input x
 graph 1- again in loop
 then x can only be deleted after the forward function has run completely
 and user doesn't need the tensor anymore */
void HPUGraph::clear_inputs() {
  // call synchronize before this API
  for (auto out_tensor : hblazy_tensors_in_out_) {
    PT_HPUGRAPH_DEBUG(
        "Freeing input-output tensor with id: ",
        out_tensor.getTensorUniqueId());
    out_tensor.SetTensorDataNullOpt();
  }
}

void HPUGraph::replay(bool async) {
  PT_LAZY_TRACE;
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  PT_HPUGRAPH_DEBUG("Replay Async = ", async)
  if (capturing_ == true) {
    // if capturing is in progress, replay is not allowed.
    PT_DEVICE_FATAL("GRAPH:: Capture in progress");
    return;
  }

  PT_IRGRAPH_DEBUG("step marker due to HPUGraph::replay");
  if (async && GET_ENV_FLAG_NEW(PT_HPU_ENABLE_HPUGRAPH_THREAD)) {
    habana_lazy::HbLazyTensor::StepMarker({}, nullptr, {}, true);
  } else {
    habana_lazy::HbLazyTensor::StepMarker({});
  }
  for (size_t i = 0; i < captured_graphs.size(); i++) {
    captured_graphs[i]->replay(async);
  }
}

void HPUGraph::replayV2(
    std::vector<at::Tensor>& static_inputs,
    std::vector<at::Tensor>& inputs,
    bool async) {
  PT_LAZY_TRACE;
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (capturing_ == true) {
    // if capturing is in progress, replay is not allowed.
    PT_DEVICE_FATAL("GRAPH:: Capture in progress");
    return;
  }

  PT_IRGRAPH_DEBUG("step marker due to HPUGraph::replayV2");
  if (async && GET_ENV_FLAG_NEW(PT_HPU_ENABLE_HPUGRAPH_THREAD)) {
    habana_lazy::HbLazyTensor::StepMarker({}, nullptr, {}, true);
  } else {
    habana_lazy::HbLazyTensor::StepMarker({});
  }

  // Use replaytV2 for the first captured graph, as the user input is
  // for the first captured graph
  if (captured_graphs.size() == 0)
    return;
  captured_graphs[0]->replayV2(static_inputs, inputs, async);

  for (size_t i = 1; i < captured_graphs.size(); i++) {
    captured_graphs[i]->replay(async);
  }
}

std::unordered_set<int64_t> get_hb_base_tensor_id_list_if_view(
    const habana_lazy::HbLazyTensor& hl_t_in) {
  // handle multi level views
  auto hl_t = hl_t_in;
  std::unordered_set<int64_t> view_t_list;
  view_t_list.insert(hl_t.getTensorUniqueId());
  while (hl_t.getDataPtr()->stride_params.has_value()) {
    auto out = hl_t.getDataPtr()->stride_params.value().base;
    hl_t = habana_lazy::GetHbLazyTensor(out, true, false);
    view_t_list.insert(hl_t.getTensorUniqueId());
  }
  return view_t_list;
}

void HPUGraph::mark_user_outputs(std::vector<at::Tensor>& outputs) {
  PT_LAZY_TRACE;
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  PT_HPUGRAPH_DEBUG("mark_user_outputs with outputs size = ", outputs.size());
  if (capturing_ == true) {
    // if capturing is in progress, replay is not allowed.
    PT_DEVICE_FATAL("GRAPH:: Capture in progress");
    return;
  }

  if (captured_graphs.empty()) {
    return;
  }

  std::unordered_set<int64_t> user_out_hblazy_tid_set;
  for (size_t graphIdx = 0; graphIdx < captured_graphs.size(); graphIdx++) {
    auto single_graph = captured_graphs[graphIdx];
    HABANA_ASSERT(
        ((single_graph->prev_graph_interdep_out_t_list_.size() == 0) &&
         (single_graph->user_out_indices_tlist_.size() == 0)),
        "Error:mark_user_outputs is called more than once for the same graph?");
    for (auto& out_tensor : single_graph->hblazy_tensors_out_) {
      user_out_hblazy_tid_set.emplace(out_tensor.getTensorUniqueId());
    }
  }

  // Go over all captured SingleHPUGraphs
  for (size_t graphIdx = 0; graphIdx < captured_graphs.size(); graphIdx++) {
    auto single_graph = captured_graphs[graphIdx];
    if (single_graph->graph_) {
      // This set shows have the indices of user_output tensors in
      // hblazy_tensors_out_
      std::unordered_set<size_t> user_out_tensors_idx_set;

      // Find all the lazy tensor id of inputs
      std::unordered_set<int64_t> input_lazyt_id_set;
      for (const auto& in_t : single_graph->hblazy_tensors_in_) {
        input_lazyt_id_set.emplace(in_t.getTensorUniqueId());
      }

      // Find the interdependant tensors
      auto out_pos = 0;
      for (auto& t : outputs) {
        size_t idx = 0;
        auto user_out_hbl_t = habana_lazy::GetHbLazyTensor(t);
        auto base_view_tids =
            get_hb_base_tensor_id_list_if_view(user_out_hbl_t);
        [[maybe_unused]] auto& ir_value = user_out_hbl_t.CurrentIrValue();
        for (auto& out_tensor : single_graph->hblazy_tensors_out_) {
          auto isSameHbTensor = out_tensor.getTensorUniqueId() ==
              user_out_hbl_t.getTensorUniqueId();

          if ((isSameHbTensor) ||
              isExists(base_view_tids, out_tensor.getTensorUniqueId())) {
            // This is used for replay to match the user out tensor indices
            single_graph->user_out_indices_tlist_.emplace_back(
                std::make_pair(out_pos, idx));
            // This is used later during replay
            user_out_tensors_idx_set.insert(idx);
          }
          idx++;
        }
        out_pos++;
      }

      // Go over all outputs from the current SingleHPUGraph
      for (size_t outIdx = 0; outIdx < single_graph->hblazy_tensors_out_.size();
           outIdx++) {
        auto& out_tensor = single_graph->hblazy_tensors_out_[outIdx];

        // exclude view tensors &  Inplace tensors
        bool isViewTensor = false;
        bool isInputTensor = false;
        if ((out_tensor.getDataPtr()->stride_params.has_value()) ||
            (out_tensor.IsCollective())) {
          isViewTensor = true;
        }

        if (isExists(input_lazyt_id_set, out_tensor.getTensorUniqueId()) &&
            !isExists(user_out_tensors_idx_set, outIdx)) {
          hblazy_tensors_in_out_.emplace_back(out_tensor);
          isInputTensor = true;
          PT_BRIDGE_DEBUG(
              "Graph: ",
              graphIdx,
              " Input found for output id: ",
              out_tensor.getTensorUniqueId());
        }

        // Check if any of the following SingleHPUGraphs use this output as an
        // input
        size_t last_use = 0;
        bool is_inter_dependent = false;
        // if its not an useroutput or if its not inplace
        if (!isInputTensor && !isViewTensor &&
            !isExists(user_out_tensors_idx_set, outIdx)) {
          for (size_t j = graphIdx + 1; j < captured_graphs.size(); j++) {
            auto next_graph = captured_graphs[j];
            // Go over all the inputs in the following SingleHPUGraph
            for (auto& hbt_in : next_graph->hblazy_tensors_in_) {
              // Found a match
              if (hbt_in.getTensorUniqueId() ==
                  out_tensor.getTensorUniqueId()) {
                is_inter_dependent = true;
                last_use = j;
                break;
              }
            }
          }
        }

        // If this tensor is used within the subgraphs, note the
        // details, this will be used during replay
        if (last_use > 0) {
          captured_graphs[last_use]->prev_graph_interdep_out_t_list_.push_back(
              out_tensor);
        }

        // Can start freeing memory for output tensors that have no dependency.
        // SetHpuGraphOutTensor mark to false so that next replay it can be
        // freed
        // Or if its an all_reduce output
        if (!isInputTensor && !isViewTensor && !is_inter_dependent &&
            !isExists(user_out_tensors_idx_set, outIdx)) {
          out_tensor.SetHpuGraphOutTensor(false);
          out_tensor.SetTensorDataNullOpt();
        }
      }

      // Remove the output vals (used to check if it's an inplace node or not)
      single_graph->output_vals_.clear();
    }
  }
}

void HPUGraph::replayV3(std::vector<at::Tensor>& inputs, bool async) {
  PT_LAZY_TRACE;
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  PT_HPUGRAPH_DEBUG(
      "replayV3 with inputs size = ", inputs.size(), " aysnc = ", async);
  if (capturing_ == true) {
    // if capturing is in progress, replay is not allowed.
    PT_DEVICE_FATAL("GRAPH:: Capture in progress");
    return;
  }

  PT_IRGRAPH_DEBUG("step marker due to HPUGraph::replayV3");
  if (async && GET_ENV_FLAG_NEW(PT_HPU_ENABLE_HPUGRAPH_THREAD)) {
    habana_lazy::HbLazyTensor::StepMarker({}, nullptr, {}, true);
  } else {
    habana_lazy::HbLazyTensor::StepMarker({});
  }

  if (captured_graphs.size() == 0) {
    return;
  }

  size_t index = 0;
  // If user called mark_user_inputs, then check if sizes in replay are same.
  for (const auto& user_input_size : user_input_sizes_) {
    if (user_input_size != inputs.at(index++).sizes().vec()) {
      PT_DEVICE_FATAL(
          "HPU GRAPH:: Mark User Input Sizes is not same Replay Input Sizes");
    }
  }

  for (size_t i = 0; i < captured_graphs.size(); i++) {
    captured_graphs[i]->replayV3(inputs, async);
  }
}

void HPUGraph::mark_user_inputs(std::vector<at::Tensor>& static_inputs) {
  PT_LAZY_TRACE;
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  PT_HPUGRAPH_DEBUG(
      "mark_user_inputs with static_inputs size = ", static_inputs.size());
  if (capturing_ == false) {
    // if capturing is not in progress, mark_user_inputs is not allowed.
    PT_DEVICE_FATAL(
        "GRAPH:: mark_user_inputs must be while capturing in progress");
    return;
  }

  habana_lazy::HbExecutionContext* context =
      habana_lazy::get_device_lazy_execution_context();
  context->setMarkedInputs(static_inputs);
  for (const auto& t : static_inputs) {
    user_input_sizes_.push_back(t.sizes().vec());
  }
}

HPUGraph::~HPUGraph() {
  habana_lazy::HbExecutionContext* context =
      habana_lazy::get_device_lazy_execution_context();
  if (context != nullptr) {
    /* Set graph capture mode off */
    context->setCapturing(false);
    context->setCaptureGraph(nullptr);
  }
}

SingleHPUGraph::~SingleHPUGraph() {
  graph_.reset();
  input_vals_.clear();
  output_vals_.clear();
  hblazy_tensors_in_.clear();
  hblazy_tensors_out_.clear();
  prev_graph_interdep_out_t_list_.clear();
  user_out_indices_tlist_.clear();
  seed_tensors_generator_.clear();
}

void SingleHPUGraph::replayGraph(
    habana_lazy::ir::ValueList& input_vals,
    bool async) {
  PT_LAZY_TRACE;
  bool dynamic_env_ = habana_helpers::GetRefineDynamicShapeStatus();
  if (dynamic_env_) {
    habana_helpers::DisableRefineDynamicShape();
  }
  habana_lazy::HbExecutionContext* context =
      habana_lazy::get_device_lazy_execution_context();

  size_t launch_jobid = context->GetUniqueJobId();
  context->AddToJobidStreamidMap(
      launch_jobid, c10::hpu::getCurrentHPUStream().stream());

  // set exec for input/output tensors
  bool queue_in_thread_pool = async &&
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_HPUGRAPH_THREAD) &&
      GET_ENV_FLAG_NEW(PT_HPU_QUEUE_SYNLAUNCHES);
  if (!queue_in_thread_pool) {
    std::unordered_set<size_t> in_uid;
    for (const auto& t : hblazy_tensors_in_) {
      auto uid = t.getTensorUniqueId();
      in_uid.insert(uid);
      t.SetExecutionInProgress();
    }

    for (auto& t : hblazy_tensors_out_) {
      auto uid = t.getTensorUniqueId();
      if (in_uid.find(uid) == in_uid.end()) {
        t.SetTensorDataNullOpt();
      }
      t.SetExecutionInProgress();
    }
  } else {
    for (const auto& t : hblazy_tensors_in_) {
      t.SetExecutionInProgress();
    }

    for (const auto& t : hblazy_tensors_out_) {
      t.SetExecutionInProgress();
    }
  }

  std::shared_ptr<habana::RecipeArgumentSpec> cached_rarg_psh = nullptr;
  if (GET_ENV_FLAG_NEW(PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK)) {
    cached_rarg_psh = cached_rarg_psh_;
  }

  if (queue_in_thread_pool) {
    context->m_launch_thread_handle =
        habana_lazy::SingleTonExecThreadPool::getInstance().enqueue(
            habana_lazy::HbLazyTensor::ExecuteCachedGraph,
            cached_rarg_psh,
            graph_,
            hash_,
            graphKey_,
            opStrs_,
            hblazy_tensors_in_,
            hblazy_tensors_out_,
            prev_graph_interdep_out_t_list_,
            seed_tensors_generator_,
            launch_jobid);
  } else {
    habana_lazy::HbLazyTensor::ExecuteCachedGraph(
        cached_rarg_psh,
        graph_,
        hash_,
        graphKey_,
        opStrs_,
        hblazy_tensors_in_,
        hblazy_tensors_out_,
        prev_graph_interdep_out_t_list_,
        seed_tensors_generator_,
        launch_jobid);
  }

  auto num_inputs = input_vals.size();
  for (size_t i = 0; i < num_inputs; ++i) {
    if (user_input_indices_.count(i) > 0) {
      hblazy_tensors_in_[i] = habana_lazy::HbLazyTensor();
      input_vals_[i] = habana_lazy::ir::Value();
    }
  }

  // Not enabling DS back once HPU graph detected
  /*if (dynamic_env_) {
    habana_helpers::EnableRefineDynamicShape();
  }*/
}

void SingleHPUGraph::replay(bool async) {
  if (graph_) {
    return replayGraph(input_vals_, async);
  }
}

void SingleHPUGraph::replayV3(std::vector<at::Tensor>& inputs, bool async) {
  PT_HPUGRAPH_DEBUG(
      "In HPUGraph::replayV3 with ", inputs.size(), " input tensors");
  PT_DEVICE_DEBUG(graph_ ? (graph_->dump(), "") : "null graph");
  if (graph_) {
    auto num_inputs = input_vals_.size();
    for (size_t i = 0; i < num_inputs; ++i) {
      if (user_input_indices_.count(i) > 0) {
        auto t = inputs[user_input_indices_[i]];
        auto hbl = habana_lazy::GetHbLazyTensor(t);
        if (!hbl.getDataPtr()->tensor_data) {
          auto& stride_params_opt = hbl.getDataPtr()->stride_params;
          if (stride_params_opt.has_value()) {
            hbl = habana_lazy::GetHbLazyTensor(
                habana_lazy::HbLazyTensorViews::HandleViewsD2H(t));
          } else {
            TORCH_CHECK(
                0, "Neither storage attached to input tensor, not its view.")
          }
        }
        hblazy_tensors_in_[i] = hbl;
        input_vals_[i] = hbl.CurrentIrValue();
      }
    }
    return replayGraph(input_vals_, async);
  }
}

void SingleHPUGraph::replayV2(
    std::vector<at::Tensor>& static_inputs,
    std::vector<at::Tensor>& inputs,
    bool async) {
  PT_HPUGRAPH_DEBUG(
      "In HPUGraph::replayV2 with ", inputs.size(), " input tensors");
  PT_DEVICE_DEBUG(graph_ ? (graph_->dump(), "") : "null graph");
  if (graph_) {
    habana_lazy::ir::ValueList input_val_list;
    std::vector<habana_lazy::HbLazyTensor> static_input_lazy_tensors;
    for (auto& t : static_inputs) {
      static_input_lazy_tensors.emplace_back(habana_lazy::GetHbLazyTensor(t));
    }
    // Input arguments:
    //  - New inputs are provided against Static inputs
    //    -> Replace the static inputs with the new inputs
    //  - No new inputs provided against static inputs (possibly weights)
    //    -> Reuse static inputs
    //  - Scalar inputs
    //    -> Reuse static scalar inputs
    // Extract the input value list from user provided input arguments
    HABANA_ASSERT(static_inputs.size() == inputs.size());
    std::transform(
        input_vals_.begin(),
        input_vals_.end(),
        std::back_inserter(input_val_list),
        [&static_input_lazy_tensors,
         &inputs](habana_lazy::ir::Value saved_ir_v_) {
          size_t idx = 0;
          for (auto& static_lazy_t : static_input_lazy_tensors) {
            // TBD: HPU graph needs to eliminate dependency on ir values
            auto& ir_value = static_lazy_t.CurrentIrValue();
            if (ir_value == saved_ir_v_) {
              return habana_lazy::GetHbLazyTensor(inputs[idx]).CurrentIrValue();
            }
            ++idx;
          }
          return saved_ir_v_;
        });
    return replayGraph(input_val_list, async);
  }
}
} // namespace hpu
} // namespace at
