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

namespace at {
namespace hpu {

HPUGraph::HPUGraph()
    // HPUStreams may not be default-constructed.
    : capture_stream_(c10::hpu::getCurrentHPUStream()) {}

void HPUGraph::capture_begin() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (capturing_) {
    // already captured started, error. only one graph
    // capture is suported.
    PT_DEVICE_FATAL("GRAPH:: graph Capture already in progress");
    return;
  }
  auto stream = c10::hpu::getCurrentHPUStream();
  capture_stream_ = stream;
  /*flush current Accumulated graph, before capture */
  habana_lazy::HbLazyTensor::StepMarkerBind("");
  capturing_ = true;
}

void HPUGraph::capture_end() {
  std::unique_lock<std::mutex> lock(mutex_);
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

  /*flush graph to capture in the end */
  habana_lazy::HbLazyTensor::StepMarkerBind("");
  habana_lazy::HbExecutionContext* context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device.id());
  graph_ = context->getGraph();
  input_vals_ = context->getInputs();
  output_vals_ = context->getOutputs();
  hblazy_tensors_ = context->getHbLazyTensors();
  capturing_ = false;
}

void HPUGraph::replay() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (capturing_ == true) {
    // if capturing is in progress, replay is not allowed.
    PT_DEVICE_FATAL("GRAPH:: Capture in progress");
    return;
  }
  if (graph_) {
    habana_lazy::HbLazyTensor::ExecuteCachedGraph(
        graph_, input_vals_, output_vals_, hblazy_tensors_, true);
  }
}

HPUGraph::~HPUGraph() {
  graph_.reset();
  input_vals_.clear();
  output_vals_.clear();
  hblazy_tensors_.clear();
}

} // namespace hpu
} // namespace at
