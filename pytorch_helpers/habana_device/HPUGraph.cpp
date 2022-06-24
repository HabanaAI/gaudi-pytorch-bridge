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
  auto stream = c10::hpu::getCurrentHPUStream();
  capture_stream_ = stream;
  /*flush current Accumulated graph, before capture */
  habana_lazy::HbLazyTensor::StepMarkerBind("");
}

void HPUGraph::capture_end() {
  auto stream = c10::hpu::getCurrentHPUStream();

  TORCH_CHECK(
      stream == capture_stream_,
      "Capture must end on the same stream it began on.");

  /*flush graph to capture in the end */
  habana_lazy::HbLazyTensor::StepMarkerBind("");
}

void HPUGraph::replay() {
  habana_lazy::HbLazyTensor::RunSavedGraph("");
}

HPUGraph::~HPUGraph() {}

} // namespace hpu
} // namespace at
