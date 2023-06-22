/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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
#include "HPUGraph.h"

#include "habana_lazy/lazy_executor.h"

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

void HPUGraph::capture_begin() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (capturing_) {
    // already captured started, error. only one graph
    // capture is suported.
    PT_DEVICE_FATAL("GRAPH:: graph Capture already in progress");
    return;
  }
  auto stream = c10::hpu::getCurrentHPUStream();
  auto& device = habana::HPURegistrar::get_device();
  habana_lazy::HbExecutionContext* context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device.id());
  capture_stream_ = stream;
  /*flush current Accumulated graph, before capture */
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
  auto& device = habana::HPURegistrar::get_device();

  habana_lazy::HbExecutionContext* context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device.id());

  /*flush graph to capture in the end */
  habana_lazy::HbLazyTensor::StepMarker({});
  capturing_ = false;

  /* Set graph capture mode off */
  context->setCapturing(false);
  context->setCaptureGraph(nullptr);

  // Save all input lazy tensors and free the output IR values
  for (size_t i = 0; i < captured_graphs.size(); i++) {
    auto single_graph = captured_graphs[i];
    auto num_inputs = single_graph->input_vals_.size() +
        single_graph->user_input_indices_.size();
    size_t saved_input_idx = 0;
    for (size_t inp = 0; inp < num_inputs; ++inp) {
      if (single_graph->user_input_indices_.count(inp) > 0) {
        single_graph->hblazy_tensors_in_.emplace_back(
            habana_lazy::HbLazyTensor());
      } else {
        std::shared_ptr<habana_lazy::Data> d =
            single_graph->input_vals_[saved_input_idx++].m_data_ptr.lock();
        single_graph->hblazy_tensors_in_.emplace_back(
            habana_lazy::HbLazyTensor(std::move(d)));
      }
    }
  }

  // Clear the user marked inputs list
  context->ClearHPUGraphUserMarkedInputs();

  // Not enabling DS back once HPU graph detected
  /*if (dynamic_env_) {
    habana_helpers::EnableRefineDynamicShape();
  }*/
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
  auto& device = habana::HPURegistrar::get_device();

  habana_lazy::HbExecutionContext* context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device.id());

  context->JoinPendingLaunchThread();
  auto captured_graph = std::make_shared<SingleHPUGraph>(
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
  PT_IRGRAPH_DEBUG("GRAPH:: captured graph");
  PT_IRGRAPH_DEBUG(
      (captured_graph->graph_ ? (captured_graph->graph_->dump(), "")
                              : "null graph"));
  PT_DEVICE_DEBUG(
      "GRAPH:: captured input size ", captured_graph->input_vals_.size());
  PT_DEVICE_DEBUG(
      "GRAPH:: captured output size ", captured_graph->output_vals_.size());
  PT_DEVICE_DEBUG(
      "GRAPH:: captured hblazy_tensors_out_ size ",
      captured_graph->hblazy_tensors_out_.size());
  context->getSeedTensorMap().clear();
  context->resetGraph();
}

void HPUGraph::replay(bool async) {
  PT_LAZY_TRACE;
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (capturing_ == true) {
    // if capturing is in progress, replay is not allowed.
    PT_DEVICE_FATAL("GRAPH:: Capture in progress");
    return;
  }

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

void HPUGraph::mark_user_outputs(std::vector<at::Tensor>& outputs) {
  PT_LAZY_TRACE;
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (capturing_ == true) {
    // if capturing is in progress, replay is not allowed.
    PT_DEVICE_FATAL("GRAPH:: Capture in progress");
    return;
  }

  if (captured_graphs.empty()) {
    return;
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
        auto& ir_value = habana_lazy::GetHbLazyTensor(t).CurrentIrValue();
        for (auto& out_tensor : single_graph->hblazy_tensors_out_) {
          auto isSameHbTensor = out_tensor.getTensorUniqueId() ==
              habana_lazy::GetHbLazyTensor(t).getTensorUniqueId();

          if (isSameHbTensor) {
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
        bool isInplaceOutTensor = false;
        if ((out_tensor.getDataPtr()->stride_params.has_value()) ||
            (single_graph->output_vals_[outIdx].IsInplace()) ||
            (isExists(input_lazyt_id_set, out_tensor.getTensorUniqueId()))) {
          isInplaceOutTensor = true;
        }

        // Check if any of the following SingleHPUGraphs use this output as an
        // input
        size_t last_use = 0;
        bool is_inter_dependent = false;
        // if its not an useroutput or if its not inplace
        if (!isInplaceOutTensor &&
            !isExists(user_out_tensors_idx_set, outIdx)) {
          for (size_t j = graphIdx + 1; j < captured_graphs.size(); j++) {
            auto next_graph = captured_graphs[j];
            size_t input_idx = 0;
            // Go over all the inputs in the following SingleHPUGraph
            for (auto& hbt_in : next_graph->hblazy_tensors_in_) {
              // Found a match
              if (hbt_in.getTensorUniqueId() ==
                  out_tensor.getTensorUniqueId()) {
                is_inter_dependent = true;
                last_use = j;
                break;
              }
              ++input_idx;
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
        if (!isInplaceOutTensor && !is_inter_dependent &&
            !isExists(user_out_tensors_idx_set, outIdx)) {
          out_tensor.SetHpuGraphOutTensor(false);
          out_tensor.SetTensorDataNullOpt();
        }
      }

      // Delete the tensors which are interdependent among multiple single
      // graphs
      for (auto& hl_t : single_graph->prev_graph_interdep_out_t_list_) {
        hl_t.SetTensorDataNullOpt();
      }

      // Remove the output vals (used to check if it's an inplace node or not)
      single_graph->output_vals_.clear();
    }
  }
}

void HPUGraph::replayV3(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    bool async) {
  PT_LAZY_TRACE;
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (capturing_ == true) {
    // if capturing is in progress, replay is not allowed.
    PT_DEVICE_FATAL("GRAPH:: Capture in progress");
    return;
  }

  if (async && GET_ENV_FLAG_NEW(PT_HPU_ENABLE_HPUGRAPH_THREAD)) {
    habana_lazy::HbLazyTensor::StepMarker({}, nullptr, {}, true);
  } else {
    habana_lazy::HbLazyTensor::StepMarker({});
  }

  if (captured_graphs.size() == 0) {
    return;
  }

  for (size_t i = 0; i < captured_graphs.size(); i++) {
    if (captured_graphs[i]->graph_) {
      for (const auto& [userOutputIdx, tensorIdx] :
           captured_graphs[i]->user_out_indices_tlist_) {
        auto& t = outputs[userOutputIdx];
        // This index must be an output tensor
        HABANA_ASSERT(
            captured_graphs[i]
                ->hblazy_tensors_out_[tensorIdx]
                .IsHpuGraphOutTensor() == true);
        auto hb_lt = habana_lazy::GetHbLazyTensor(t);
        hb_lt.SetHpuGraphOutTensor(true);
        captured_graphs[i]->hblazy_tensors_out_[tensorIdx] = hb_lt;
      }
    }
    captured_graphs[i]->replayV3(outputs, inputs, async);
  }
}

void HPUGraph::mark_user_inputs(std::vector<at::Tensor>& static_inputs) {
  PT_LAZY_TRACE;
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (capturing_ == false) {
    // if capturing is not in progress, mark_user_inputs is not allowed.
    PT_DEVICE_FATAL(
        "GRAPH:: mark_user_inputs must be while capturing in progress");
    return;
  }

  auto& device = habana::HPURegistrar::get_device();

  habana_lazy::HbExecutionContext* context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device.id());
  context->setMarkedInputs(static_inputs);
}

HPUGraph::~HPUGraph() {
  auto& device = habana::HPURegistrar::get_device();

  habana_lazy::HbExecutionContext* context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device.id());
  /* Set graph capture mode off */
  context->setCapturing(false);
  context->setCaptureGraph(nullptr);
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
  bool dynamic_env_ = habana_helpers::GetRefineDynamicShapeStatus();
  if (dynamic_env_) {
    habana_helpers::DisableRefineDynamicShape();
  }

  auto& device = habana::HPURegistrar::get_device();
  habana_lazy::HbExecutionContext* context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device.id());

  size_t launch_jobid = context->GetUniqueJobId();
  context->AddToJobidStreamidMap(
      launch_jobid, c10::hpu::getCurrentHPUStream().stream());

  // set exec for input/output tensors
  for (const auto& t : hblazy_tensors_in_) {
    t.SetExecutionInProgress();
  }

  for (const auto& t : hblazy_tensors_out_) {
    t.SetExecutionInProgress();
  }

  if (async && GET_ENV_FLAG_NEW(PT_HPU_ENABLE_HPUGRAPH_THREAD) &&
      GET_ENV_FLAG_NEW(PT_HPU_QUEUE_SYNLAUNCHES)) {
    context->m_launch_thread_handle =
        habana_lazy::SingleTonExecThreadPool::getInstance().enqueue(
            habana_lazy::HbLazyTensor::ExecuteCachedGraph,
            graph_,
            hash_,
            graphKey_,
            opStrs_,
            hblazy_tensors_in_,
            hblazy_tensors_out_,
            prev_graph_interdep_out_t_list_,
            seed_tensors_generator_,
            true /*is_cached*/,
            launch_jobid);
  } else {
    habana_lazy::HbLazyTensor::ExecuteCachedGraph(
        graph_,
        hash_,
        graphKey_,
        opStrs_,
        hblazy_tensors_in_,
        hblazy_tensors_out_,
        prev_graph_interdep_out_t_list_,
        seed_tensors_generator_,
        true /*is_cached*/,
        launch_jobid);
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

void SingleHPUGraph::replayV3(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    bool async) {
  PT_DEVICE_DEBUG(
      "In HPUGraph::replayV3 with ", outputs.size(), " output tensors");
  PT_DEVICE_DEBUG(graph_ ? (graph_->dump(), "") : "null graph");
  if (graph_) {
    auto num_inputs = input_vals_.size() + user_input_indices_.size();
    habana_lazy::ir::ValueList input_val_list;
    size_t saved_input_idx = 0;
    for (size_t i = 0; i < num_inputs; ++i) {
      if (user_input_indices_.count(i) > 0) {
        input_val_list.emplace_back(
            habana_lazy::GetHbLazyTensor(inputs[user_input_indices_[i]])
                .CurrentIrValue());
        hblazy_tensors_in_[i] =
            habana_lazy::GetHbLazyTensor(inputs[user_input_indices_[i]]);
      } else {
        input_val_list.emplace_back(input_vals_[saved_input_idx]);
        ++saved_input_idx;
      }
    }
    return replayGraph(input_val_list, async);
  }
}

void SingleHPUGraph::replayV2(
    std::vector<at::Tensor>& static_inputs,
    std::vector<at::Tensor>& inputs,
    bool async) {
  PT_DEVICE_DEBUG(
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
