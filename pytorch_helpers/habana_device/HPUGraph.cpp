/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "HPUGraph.h"

#include "habana_lazy/lazy_executor.h"

namespace at {
namespace hpu {

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
  auto& device = synapse_helpers::HPURegistrar::get_device();
  habana_lazy::HbExecutionContext* context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device.id());
  capture_stream_ = stream;
  /*flush current Accumulated graph, before capture */
  habana_lazy::HbLazyTensor::StepMarkerBind("");
  context->JoinPendingLaunchThread();

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
  auto& device = synapse_helpers::HPURegistrar::get_device();

  habana_lazy::HbExecutionContext* context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device.id());

  /*flush graph to capture in the end */
  habana_lazy::HbLazyTensor::StepMarkerBind("");
  capturing_ = false;

  /* Set graph capture mode off */
  context->setCapturing(false);
  context->setCaptureGraph(nullptr);
  if (dynamic_env_) {
    habana_helpers::EnableRefineDynamicShape();
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
  auto& device = synapse_helpers::HPURegistrar::get_device();

  habana_lazy::HbExecutionContext* context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device.id());

  context->JoinPendingLaunchThread();
  auto captured_graph = std::make_shared<SingleHPUGraph>(
      context->getGraph(),
      context->getInputs(),
      context->getOutputs(),
      context->getHbLazyTensors(),
      context->getSeedTensorMap(),
      context->getHash(),
      context->getGraphKey(),
      context->getOpStrs());
  captured_graphs.push_back(captured_graph);
  PT_IRGRAPH_DEBUG("GRAPH:: captured graph ");
  PT_IRGRAPH_DEBUG(
      (captured_graph->graph_ ? (captured_graph->graph_->dump(), "")
                              : "null graph"));
  PT_DEVICE_DEBUG(
      "GRAPH:: captured input size ", captured_graph->input_vals_.size());
  PT_DEVICE_DEBUG(
      "GRAPH:: captured output size ", captured_graph->output_vals_.size());
  PT_DEVICE_DEBUG(
      "GRAPH:: captured hblazy_tensors_ size ",
      captured_graph->hblazy_tensors_.size());
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

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_HPUGRAPH_THREAD)) {
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

HPUGraph::~HPUGraph() {
  auto& device = synapse_helpers::HPURegistrar::get_device();

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
  hblazy_tensors_.clear();
  seed_tensors_generator_.clear();
}

void SingleHPUGraph::replayGraph(
    habana_lazy::ir::ValueList& input_vals,
    bool async) {
  bool dynamic_env_ = habana_helpers::GetRefineDynamicShapeStatus();
  if (dynamic_env_) {
    habana_helpers::DisableRefineDynamicShape();
  }

  if (async && GET_ENV_FLAG_NEW(PT_HPU_ENABLE_HPUGRAPH_THREAD) &&
      GET_ENV_FLAG_NEW(PT_HPU_QUEUE_SYNLAUNCHES) &&
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LAUNCHTHREAD_USE_THREADPOOL)) {
    auto& device = synapse_helpers::HPURegistrar::get_device();
    habana_lazy::HbExecutionContext* context =
        habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
            device.id());

    context->m_launch_thread_handle =
        habana_lazy::SingleTonExecThreadPool::getInstance().enqueue(
            habana_lazy::HbLazyTensor::ExecuteCachedGraph,
            graph_,
            hash_,
            graphKey_,
            opStrs_,
            input_vals,
            output_vals_,
            hblazy_tensors_,
            seed_tensors_generator_,
            true /*is_cached*/);
  } else {
    habana_lazy::HbLazyTensor::ExecuteCachedGraph(
        graph_,
        hash_,
        graphKey_,
        opStrs_,
        input_vals,
        output_vals_,
        hblazy_tensors_,
        seed_tensors_generator_,
        true /*is_cached*/);
  }
  if (dynamic_env_) {
    habana_helpers::EnableRefineDynamicShape();
  }
}

void SingleHPUGraph::replay(bool async) {
  if (graph_) {
    return replayGraph(input_vals_, async);
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
