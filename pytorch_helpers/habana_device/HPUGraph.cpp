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
      context->getHash(),
      context->getGraphKey(),
      context->getOpStrs());
  captured_graphs.push_back(captured_graph);
  PT_DEVICE_DEBUG("GRAPH:: captured graph ");
  PT_DEVICE_DEBUG(
      (captured_graph->graph_ ? (captured_graph->graph_->dump(), "")
                              : "null graph"));
  PT_DEVICE_DEBUG(
      "GRAPH:: captured input size ", captured_graph->input_vals_.size());
  PT_DEVICE_DEBUG(
      "GRAPH:: captured output size ", captured_graph->output_vals_.size());
  PT_DEVICE_DEBUG(
      "GRAPH:: captured hblazy_tensors_ size ",
      captured_graph->hblazy_tensors_.size());
}

void HPUGraph::replay() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (capturing_ == true) {
    // if capturing is in progress, replay is not allowed.
    PT_DEVICE_FATAL("GRAPH:: Capture in progress");
    return;
  }

  habana_lazy::HbLazyTensor::StepMarker({});
  for (size_t i = 0; i < captured_graphs.size(); i++) {
    captured_graphs[i]->replay();
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
}

void SingleHPUGraph::replay() {
  if (graph_) {
    habana_lazy::HbLazyTensor::ExecuteCachedGraph(
        graph_,
        hash_,
        graphKey_,
        opStrs_,
        input_vals_,
        output_vals_,
        hblazy_tensors_,
        true /*is_cached*/);
  }
}

} // namespace hpu
} // namespace at
